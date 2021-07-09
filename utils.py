# -*- coding: utf-8 -*-
import torch
import numpy as np
import math
import copy
from torch.utils.data import Dataset


def get_cond(cs_ratio, sigma, type):
    para_cs = None
    para_noise = sigma / 5.0
    if type == 'org':
        para_cs = cs_ratio * 2.0 / 100.0
    elif type == 'org_ratio':
        para_cs = cs_ratio / 100.0

    para_cs_np = np.array([para_cs])

    para_cs = torch.from_numpy(para_cs_np).type(torch.FloatTensor)

    para_cs = para_cs.cuda()

    para_noise_np = np.array([para_noise])
    para_noise = torch.from_numpy(para_noise_np).type(torch.FloatTensor)

    para_noise = para_noise.cuda()
    para_cs = para_cs.view(1, 1)
    para_noise = para_noise.view(1, 1)
    para = torch.cat((para_cs, para_noise), 1)


    return para


def write_data(file_name, data):
    f = open(file_name, 'a')
    f.write(data)
    f.close()


def add_test_noise(img, sigma, device, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, sigma / 255., img.shape)
    noise = torch.from_numpy(noise).float().to(device)
    img.add_(noise)
    return img


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


def rgb2ycbcr(rgb):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    return ycbcr.reshape(shape)


# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:, 0] -= 16.
    rgb[:, 1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)


def imread_CS_py(Iorg, block_size=33):
    [row, col] = Iorg.shape
    row_pad = block_size - np.mod(row, block_size)
    col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size=33):
    [row, col] = Ipad.shape
    row_block = row / block_size
    col_block = col / block_size
    block_num = int(row_block * col_block)
    img_col = np.zeros([block_size ** 2, block_num])
    count = 0
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new, block_size=33):
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
