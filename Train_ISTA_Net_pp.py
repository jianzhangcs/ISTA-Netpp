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
from utils import imread_CS_py, img2col_py, col2im_CS_py, psnr, add_test_noise, write_data,get_cond
import math
from torch.utils.data import Dataset, DataLoader
import csdata_fast
import torch.nn.functional as F


parser = ArgumentParser(description='ISTA-Net-plus')


parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=400, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=50, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--data_dir', type=str, default='cs_train400_png', help='training data directory')
parser.add_argument('--rgb_range', type=int, default=1, help='value range 1 or 255')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument('--patch_size', type=int, default=33, help='from {1, 4, 10, 25, 40, 50}')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir_org', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--ext', type=str, default='.png', help='training data directory')

parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--test_cycle', type=int, default=10, help='epoch number of each test cycle')


args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list



test_name = args.test_name
test_cycle = args.test_cycle

test_dir = os.path.join(args.data_dir_org, test_name)
filepaths = glob.glob(test_dir + '/*.tif')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}


n_input = ratio_dict[cs_ratio]
n_output = 1089
batch_size = 64


Phi_input = None
total_phi_num = 50
rand_num = 1

train_cs_ratio_set = [10, 20, 30, 40, 50]
test_cs_ratio_set = [10,20,30,40, 50]

model_name = 'ISTA_Net_pp'

Phi_all = {}
for cs_ratio in train_cs_ratio_set:
    size_after_compress = ratio_dict[cs_ratio]
    Phi_all[cs_ratio] = np.zeros((int(rand_num * 1), size_after_compress, 1089))
    Phi_name = './%s/phi_sampling_%d_%dx%d.npy' % (args.matrix_dir, total_phi_num, size_after_compress, 1089)
    Phi_data = np.load(Phi_name)

    for k in range(rand_num):
        Phi_all[cs_ratio][k, :, :] = Phi_data[k, :, :]


Qinit = None
def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)
class condition_network(nn.Module):
    def __init__(self,LayerNo):
        super(condition_network, self).__init__()

        self.fc1 = nn.Linear(1, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, LayerNo+LayerNo, bias=True)

        self.act12 = nn.ReLU(inplace=True)
        self.act3 = nn.Softplus()

    def forward(self, x):
        x=x[:,0:1]

        x = self.act12(self.fc1(x))
        x = self.act12(self.fc2(x))
        x = self.act3(self.fc3(x))
        num=x.shape[1]
        num=int(num/2)
        return x[0,0:num],x[0,num:]

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


    def forward(self, x,  PhiWeight, PhiTWeight, PhiTb,lambda_step,x_step):
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
        self.condition = condition_network(LayerNo)

    def forward(self, x, Phi, Qinit,n_input): 

        batchx = x[0]  
        cond = x[1]  
        lambda_step,x_step = self.condition(cond)
        
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(batchx, PhiWeight, padding=0, stride=33, bias=None)    # Get measurements

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb   

        for i in range(self.LayerNo):
            x = self.fcs[i](x,  PhiWeight, PhiTWeight, PhiTb,lambda_step[i],x_step[i])


        x_final = x

        return x_final

model = ISTA_Net_pp(layer_num)
model = nn.DataParallel(model)
model = model.to(device)


print_flag = 0

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())



class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


# training_data = csdata.SlowDataset(args)
training_data = csdata_fast.SlowDataset(args)


if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=8,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/%s_layer_%d_group_%d_ratio_all_lr_%.4f" % (args.model_dir, model_name, layer_num, group_num, learning_rate)

log_file_name = "./%s/%s_Log_layer_%d_group_%d_ratio_all_lr_%.4f.txt" % (args.log_dir, model_name, layer_num, group_num, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if start_epoch > 0:

    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, start_epoch)))


Phi = {}
for cs_ratio in train_cs_ratio_set:
    Phi[cs_ratio] = torch.from_numpy(Phi_all[cs_ratio]).type(torch.FloatTensor)
    Phi[cs_ratio] = Phi[cs_ratio].to(device)
cur_Phi = None  




# Training loop
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    for data in rand_loader:

        batch_x = data.view(-1, 1, 33, 33)
        batch_x = batch_x.to(device)

        rand_Phi_index = np.random.randint(rand_num * 1)

        rand_cs_ratio = np.random.choice(train_cs_ratio_set)
        cur_Phi = Phi[rand_cs_ratio][rand_Phi_index]

        n_input = ratio_dict[rand_cs_ratio]
        x_input_x = [batch_x, get_cond(rand_cs_ratio, 0.0, 'org_ratio')]
        x_output = model(x_input_x, cur_Phi, Qinit,n_input)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

        loss_all = loss_discrepancy

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

    output_data = str(datetime.now()) + " [%d/%d] Total loss: %.4f, discrepancy loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item())
    print(output_data)

    write_data(log_file_name, output_data)

    if epoch_i % 10 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters



