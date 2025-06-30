import glob
import os
import time
from collections import OrderedDict
import sys
sys.path.append('.')
import numpy as np
import torch
import cv2
import argparse
import custom_utils

from natsort import natsort
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from math import log10, sqrt
import pyiqa
from pytorch_msssim import ssim as ssim2
from IQA_pytorch import SSIM
from skimage.metrics import structural_similarity as compare_ssim
import scipy.misc
from scipy.ndimage import gaussian_filter
import lpips

import numpy as np
import skimage
from skimage import data, img_as_float
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.niqe_val = pyiqa.create_metric("niqe", device=torch.device('cuda'))
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)
        self.ssim_val = SSIM()

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips, self.mae, self.niqe]]


    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)

        return psnr_val

    def ssim(self, imgA, imgB, gray_scale=False):
        if gray_scale:
            score, diff = ssim(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True, multichannel=True)
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        else:
            score, diff = ssim(imgA, imgB, full=True, multichannel=True,channel_axis=2)
        return score

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    # 此计算方式同2023 AAAI_LLFormer_Ultra-High-Definition Low-Light Image Enhancement：A Benchmark and Transformer-Based Method
    def mae(self, imgA, imgB):

        imgA = TF.to_tensor(imgA).float()
        imgB = TF.to_tensor(imgB).float()

        # 计算差异
        diff = torch.abs(imgA - imgB)

        # 计算MAE
        mae = torch.mean(diff)

        return mae.item()


    def niqe(self, gt, enhance):
        gt = TF.to_tensor(gt).float()
        enhance = TF.to_tensor(enhance).float()
        gt, enhance = gt.unsqueeze(0), enhance.unsqueeze(0)
        score = self.niqe_val(enhance).item()
        return score


    # def niqe(self, gt, enhance):
    #     img = enhance
    #     img = img_as_float(img)
        
    #     if img.ndim == 3:
    #         img = rgb2gray(img)
        
    #     img = resize(img, (384, 384), anti_aliasing=True)
        
    #     mu = convolve(img, np.ones((7, 7)) / 49, mode='reflect')
    #     mu_sq = mu * mu
    #     sigma = np.sqrt(abs(convolve(img * img, np.ones((7, 7)) / 49, mode='reflect') - mu_sq))
    #     structdis = convolve(abs(img - mu), np.ones((7, 7)) / 49, mode='reflect')
    #     niqe_score = np.mean(sigma / (structdis + 1e-12))
        
    #     return niqe_score


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def format_result(psnr, ssim, lpips, mae, niqe):
    return f'{psnr:0.4f}, {ssim:0.4f}, {lpips:0.4f}, {mae:0.4f}, {niqe:0.4f}'

def measure_dirs(dirA, dirB, dataset_name, txt_path, use_gpu, verbose=False):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type}'))
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))

    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)

    measure = Measure(use_gpu=use_gpu)


    results = []
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()

        t = time.time()
        high = imread(pathA)
        low = imread(pathB)
        img_name = os.path.basename(pathA)

        result['psnr'], result['ssim'], result['lpips'], result['mae'], result['niqe'] = measure.measure(high, low)
        # d = time.time() - t
        # vprint(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(**result)}, {d:0.1f}")

        d = time.time() - t
        # output_str = f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]} >>\t "
        output_str = f"{pathA.split('/')[-1]} >>\t "
        output_str += f"PSNR: {result['psnr']:.2f} \t SSIM: {result['ssim']:.3f} \t LPIPS: {result['lpips']:.3f} \t MAE: {result['mae']:.3f} \t NIQE: {result['niqe']:.3f} \t "
        output_str += f"Time taken: {d:0.1f}s"
        vprint(output_str)

        results.append(result)

        with open(txt_path, 'a+') as f:
            f.write("Image: {} \t >> PSNR: {:.2f} \t SSIM: {:.4f} \t LPIPS: {:.4f} \t NIQE: {:.2f} \n".format(img_name, result['psnr'], result['ssim'], result['lpips'], result['mae'], result['niqe']))


    psnr = np.mean([result['psnr'] for result in results])
    ssim = np.mean([result['ssim'] for result in results])
    lpips = np.mean([result['lpips'] for result in results])
    mae = np.mean([result['mae'] for result in results])
    niqe = np.mean([result['niqe'] for result in results])


    with open(txt_path, 'a+') as f:
        f.write("\n------------------------------ \n\n{} \t >> PSNR: {:.2f} \t SSIM: {:.3f} \t LPIPS: {:.3f} \t MAE: {:.3f}\t NIQE: {:.3f}\n\n------------------------------".format(dataset_name, psnr, ssim, lpips, mae, niqe))

    # vprint(f"Final Result: {format_result(psnr, ssim, lpips, mae, niqe)}, {time.time() - t_init:0.1f}s")
    result = format_result(psnr, ssim, lpips, mae, niqe)
    result_names = {'PSNR': psnr, 'SSIM': ssim, 'LPIPS': lpips, 'MAE': mae, 'NIQE': niqe}

    output_str = "\nFinal Result>> \t"
    for name, value in result_names.items():
        if name == 'PSNR':
            output_str += f"{name}: {value:.2f} \t"  # 保留两位小数
        else:
            output_str += f"{name}: {value:.3f} \t"  # 保留三位小数

    output_str += f"Time taken: {time.time() - t_init:0.1f}s"
    vprint(output_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='MIT_5K', type=str)
    parser.add_argument('--model_name', default='ReTrust', type=str)
    parser.add_argument('--result_dir', default='./results', type=str)

    parser.add_argument('-type', default='png')
    parser.add_argument('--gpu_id', default=False)
    args = parser.parse_args()

    if args.dataset_name == 'LOL_v1':
        dirA = '/data/xr/Dataset/light_dataset/LOL_v1/eval15/high/'
        dirB = os.path.join('results/LOL_v1/', args.model_name)
    elif args.dataset_name == 'LOL_v2_real':
        dirA = '/data/xr/Dataset/light_dataset/LOL_v2/Real_captured/Test/high/'
        dirB = os.path.join('results/LOL_v2_real/', args.model_name)
    elif args.dataset_name == 'LOL_v2_sync':
        dirA = '/data/xr/Dataset/light_dataset/LOL_v2/Synthetic/Test/high/'
        dirB = os.path.join('results/LOL_v2_sync/', args.model_name)
    elif args.dataset_name == 'MIT_5K':
        dirA = '/data/xr/Dataset/light_dataset/MIT-Adobe-5K-512/test/high/'
        dirB = os.path.join('results/MIT_5K/', args.model_name)
    elif args.dataset_name == 'LOL_blur':
        dirA = '/data/xr/Dataset/light_dataset/LOL_blur/eval/high/'
        dirB = os.path.join('results/LOL_blur/', args.model_name)   
    elif args.dataset_name == 'SID':
        dirA = '/data/xr/Dataset/light_dataset/SID_png/eval/high/'
        dirB = os.path.join('results/SID/', args.model_name)  
    elif args.dataset_name == 'SMID':
        dirA = '/data/xr/Dataset/light_dataset/SMID_png/eval/high/'
        dirB = os.path.join('results/SMID/', args.model_name)  
    elif args.dataset_name == 'SDSD_indoor':
        dirA = '/data/xr/Dataset/light_dataset/SDSD_indoor_png/eval/high/'
        dirB = os.path.join('results/SDSD_indoor/', args.model_name)  
    elif args.dataset_name == 'SDSD_outdoor':
        dirA = '/data/xr/Dataset/light_dataset/SDSD_outdoor_png/eval/high/'
        dirB = os.path.join('results/SDSD_outdoor/', args.model_name)  



    result_path = os.path.join(args.result_dir, args.dataset_name)
    # if not os.path.exists(result_path):
    #     custom_utils.mkdir(result_path)
    txt_path = result_path + '/result.txt'

    if os.path.exists(txt_path):
        os.remove(txt_path)

    type = args.type
    use_gpu = args.gpu_id

    if len(dirA) > 0 and len(dirB) > 0:
        measure_dirs(dirA, dirB, dataset_name=args.dataset_name, txt_path=txt_path, use_gpu=use_gpu, verbose=True)
