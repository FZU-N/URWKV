
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1" #str(args.gpu_id)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# torch imports
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from collections import OrderedDict

# other imports

import sys
sys.path.append('.')
import argparse
from thop import profile
import numpy as np
from IQA_pytorch import SSIM, MS_SSIM
from tqdm import tqdm
import pyiqa
import cv2
import matplotlib as plt
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import Image

# ours imports
import custom_utils
from custom_utils.data_loaders.lol import wholeDataLoader
from model.model_builder import LLENet
from custom_utils.img_resize import pad_input, crop_output
from custom_utils.validation import PSNR, validation

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--testSet_path', type=str, default='/data/xr/Dataset/light_dataset/LOL_v1/eval15')
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--channel', type=int, default=32)
parser.add_argument('--model_name', type=str, default='testpath')
parser.add_argument('--weight_path', type=str, default='./checkpoints/LOL_v1/testpath/models/model_latest.pth')
parser.add_argument('--save_path', type=str, default='./results/LOL_v1')
args = parser.parse_args()

print(args)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    # model.load_state_dict(checkpoint["state_dict"])
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def eval():
    val_dataset = wholeDataLoader(images_path=args.testSet_path, mode='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    model = LLENet(dim=args.channel).cuda()

    test_tensor = torch.randn(1, 3, 256, 256)
    flops, params = profile(model.cuda(), ((test_tensor.cuda()),))
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    params_number = params / 1000000.0
    flops_number = flops / 1000000000.0

    # model.load_state_dict(torch.load(args.weight_path))
    load_checkpoint(model, args.weight_path)
    model.eval()


    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    lpips_list = []
    niqe_list = []

    if args.save:
        result_path = os.path.join(args.save_path, args.model_name)
        if not os.path.exists(result_path):
            custom_utils.mkdir(result_path)

    if os.path.exists(result_path + '/result.txt'):
        os.remove(result_path + '/result.txt')

    with torch.no_grad():
        for i, imgs in enumerate(tqdm(val_loader), 0):
            low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
            enhanced_img = model(low_img)

  
            # 图像上不输出相关信息
            if args.save:
                save_path = os.path.join(args.save_path, args.model_name)
                save_name = str(name) + '.png'
                save_file = os.path.join(save_path, save_name)       
                torchvision.utils.save_image(enhanced_img, save_file)

if __name__ == "__main__":

    eval()