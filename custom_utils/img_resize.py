import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

import cv2
import os
import math


def pad_input(net_input, local_window_dim=2):
    _, _, h_old, w_old = net_input.size()
    h_original = h_old
    w_original = w_old
    multiplier = max(h_old // local_window_dim + 1, w_old // local_window_dim + 1)
    h_pad = (multiplier) * local_window_dim - h_old
    w_pad = (multiplier) * local_window_dim - w_old
    net_input = torch.cat([net_input, torch.flip(net_input, [2])], 2)[:, :, :h_old + h_pad, :]
    net_input = torch.cat([net_input, torch.flip(net_input, [3])], 3)[:, :, :, :w_old + w_pad]

    if h_pad > h_old or w_pad > w_old:
        _, _, h_old, w_old = net_input.size()
        multiplier = max(h_old // local_window_dim + 1, w_old // local_window_dim + 1)
        h_pad = (multiplier) * local_window_dim - h_old
        w_pad = (multiplier) * local_window_dim - w_old
        net_input = torch.cat([net_input, torch.flip(net_input, [2])], 2)[:, :, :h_old + h_pad, :]
        net_input = torch.cat([net_input, torch.flip(net_input, [3])], 3)[:, :, :, :w_old + w_pad]

    return net_input


def crop_output(net_input, net_output):
    _, _, h_old, w_old = net_input.size()
    h_original = h_old
    w_original = w_old
    net_output = net_output[:,:,:h_original, :w_original]
    # output_data = net_output.cpu().detach().numpy() #B C H W
    # output_data = np.transpose(output_data, (0,2,3,1)) #B H W C 
    output_data = net_output

    return output_data