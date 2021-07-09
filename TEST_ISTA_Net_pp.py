# -*- coding: utf-8 -*-
import os
from datetime import datetime

import cv2
import glob
import torch
import platform
import numpy as np
from tqdm import tqdm
from time import time
import torch.nn as nn
import scipy.io as sio
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from skimage.measure import compare_ssim as ssim
import math
import torch.nn.functional as F
from utils import RandomDataset, imread_CS_py, img2col_py, col2im_CS_py, psnr, add_test_noise, write_data,get_cond

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--test_epoch', type=int, default=401, help='epoch number of start training')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--test_cycle', type=int, default=10, help='epoch number of each test cycle')


args = parser.parse_args()

test_epoch = args.test_epoch
# end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list



test_name = args.test_name
test_cycle = args.test_cycle

test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.*')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict =  {1: 10, 4: 43, 10: 109, 20: 218, 24:261,25: 272,27:294, 30: 327, 33:359,36:392,40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
batch_size = 64


Phi_input = None
total_phi_num = 50
rand_num = 1


test_cs_ratio_set = [cs_ratio]
test_sigma_set = [0.0]

model_name = 'ISTA_Net_pp'

Phi_all = {}
for cs_ratio in test_cs_ratio_set:
    size_after_compress = ratio_dict[cs_ratio]
    Phi_all[cs_ratio] = np.zeros((int(rand_num * 1), size_after_compress, 1089))
    Phi_name = './%s/phi_sampling_%d_%dx%d.npy' % (args.matrix_dir, total_phi_num, size_after_compress, 1089)
    Phi_data = np.load(Phi_name)


    for k in range(rand_num):
        Phi_all[cs_ratio][k, :, :] = Phi_data[k, :, :]



Qinit = None


class condition_network(nn.Module):
    def __init__(self):
        super(condition_network, self).__init__()

        self.fc1 = nn.Linear(1, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 40, bias=True)

        self.act12 = nn.ReLU(inplace=True)
        self.act3 = nn.Softplus()

    def forward(self, x):
        x=x[:,0:1]
        x = self.act12(self.fc1(x))
        x = self.act12(self.fc2(x))
        x = self.act3(self.fc3(x))

        return x[0,0:20],x[0,20:40]

class ResidualBlock_basic(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_basic, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = x[1]
        content = x[0]

        out = self.act(self.conv1(content))
        out = self.conv2(out)
        return content + out, cond


class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.head_conv = nn.Conv2d(2, 32, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            ResidualBlock_basic(nf=32),
            ResidualBlock_basic(nf=32)
        )
        self.tail_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

    def forward(self, x, PhiWeight, PhiTWeight, PhiTb,lambda_step,x_step):
        x = x - lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x = x + lambda_step * PhiTb

        x_input = x
        sigma= x_step.repeat(x_input.shape[0], 1, x_input.shape[2], x_input.shape[3])
        x_input_cat = torch.cat((x_input,sigma),1)
        x_mid = self.head_conv(x_input_cat)
        cond = None

        x_mid, cond = self.ResidualBlocks([x_mid, cond])
        x_mid = self.tail_conv(x_mid)

        x_pred = x_input + x_mid
        return x_pred




# Define ISTA-Net-pp
class ISTA_Net_pp(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTA_Net_pp, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.condition = condition_network()

    def forward(self, x, Phi, Qinit,n_input): 

        batch_x = x[0]  
        cond = x[1]  
        lambda_step,x_step = self.condition(cond)
        
        PhiWeight = Phi.contiguous().view(-1, 1, 33, 33)
        Phix = F.conv2d(batch_x, PhiWeight, padding=0, stride=33, bias=None)    # Get measurements

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb    # Conduct initialization

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiWeight, PhiTWeight, PhiTb,lambda_step[i],x_step[i])

        x_final = x

        return x_final

model = ISTA_Net_pp(layer_num)
model = nn.DataParallel(model)
model = model.to(device)


print_flag = 1
if print_flag:
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/%s_layer_%d_group_%d_ratio_all_lr_%.4f" % (args.model_dir, model_name, layer_num, group_num,  learning_rate)

log_file_name = "./%s/%s_Log_testset_%s_layer_%d_group_%d_ratio_%d_lr_%.4f.txt" % (args.log_dir, model_name, args.test_name,layer_num, group_num, cs_ratio, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print(model_dir)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, test_epoch)))

def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)


Phi = {}
for cs_ratio in test_cs_ratio_set:
    Phi[cs_ratio] = torch.from_numpy(Phi_all[cs_ratio]).type(torch.FloatTensor)
    Phi[cs_ratio] = Phi[cs_ratio].to(device)
cur_Phi = None 





def test_model(epoch_num, cs_ratio, sigma, model_name):
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    COST_TIME_All = np.zeros([1, ImgNum], dtype=np.float32)


    rand_Phi_index = 0
    cur_Phi = Phi[cs_ratio][rand_Phi_index]
    print("(Test)CS reconstruction start, using Phi[%d][%d] to test" % (cs_ratio, rand_Phi_index))

    with torch.no_grad():
        for img_no in tqdm(range(ImgNum)):
            imgName = filepaths[img_no]

            Img = cv2.imread(imgName, 1)

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()

            Iorg_y = Img_yuv[:, :, 0]

            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
            Img_output = Ipad.reshape(1, 1, row_new, col_new)/255.0

            start_time = time()


            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)


            n_input = ratio_dict[cs_ratio]
            x_input_x = [batch_x, get_cond(cs_ratio, sigma, 'org_ratio')]
            x_output = model(x_input_x, cur_Phi, Qinit,n_input)

            end_time = time()

            Prediction_value = x_output.cpu().data.numpy().squeeze()

            X_rec = np.clip(Prediction_value[:row,:col], 0, 1).astype(np.float64)

            rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

            print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, (end_time - start_time), rec_PSNR, rec_SSIM))

            Img_rec_yuv[:,:,0] = X_rec*255

            im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

            resultName = imgName.replace(args.data_dir, args.result_dir)
            cv2.imwrite("%s_%s_layer_%d_ratio_%d_sigma_%d_lr_%.4f_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (
                resultName, model_name, layer_num, cs_ratio, sigma, learning_rate, epoch_num, rec_PSNR, rec_SSIM),
                        im_rec_rgb)

            del x_output

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            COST_TIME_All[0, img_no] = end_time - start_time

    print_data = str(
        datetime.now()) + " CS ratio is %d, avg PSNR/SSIM for %s is %.2f/%.4f, epoch number of model is %d, avg cost time is %.4f second(s)\n" % (
                 cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num, np.mean(COST_TIME_All))
    print(print_data)

    output_file_name = "./%s/%s_Results_testset_%s_layer_%d_group_%d_ratio_%d_sigma_%d_lr_%.4f.txt" % (
    args.log_dir, model_name,args.test_name, layer_num, group_num, cs_ratio, sigma, learning_rate)

    output_data = "%d, %.2f, %.4f, %.4f\n" % (epoch_num, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(COST_TIME_All))

    write_data(output_file_name, output_data)  





for cs_ratio in test_cs_ratio_set:
    for test_sigma in test_sigma_set:
        test_model(test_epoch, cs_ratio, test_sigma, model_name)
