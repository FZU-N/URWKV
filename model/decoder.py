# troch imports
import torch
from torch import nn
from torchvision import datasets, transforms
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

# other imports 
import numpy as np
import os
import math
from .modules.SSF import SSF
from .modules.urwkv import URWKV

class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.patch_size = 4
        self.residual_depth = [1, 1, 1]
        self.recursive_depth = [1, 1, 1]
        self.enhanceBlock1 = URWKV(patch_size=3, in_channels=self.dim*4, embed_dims=self.dim*4, depth=2)
        self.proj1 = nn.Sequential(
            nn.Conv2d(self.dim*4, self.dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        

        self.enhanceBlock2 = URWKV(patch_size=3, in_channels=self.dim*2, embed_dims=self.dim*2, depth=2)
        self.proj2 = nn.Sequential(
            nn.Conv2d(self.dim*2, self.dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        

        self.enhanceBlock3 = URWKV(patch_size=3, in_channels=self.dim, embed_dims=self.dim, depth=2)
        self.proj3 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.tail = nn.Conv2d(self.dim, 3, kernel_size=3, stride=1, padding=1, bias=False)


        self.multiscale_fuse1 = SSF(num_feats=3, encode_channels=[self.dim*4, self.dim*2, self.dim], target_channels=self.dim)
        self.multiscale_fuse2 = SSF(num_feats=3, encode_channels=[self.dim*4, self.dim*2, self.dim], target_channels=self.dim*2)
        self.multiscale_fuse3 = SSF(num_feats=3, encode_channels=[self.dim*4, self.dim*2, self.dim], target_channels=self.dim*4)

        self.upSample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x, encode_list, inter_feat):
        feat_1s, feat_2s, feat_4s = encode_list[0], encode_list[1], encode_list[2]
        
        
        x1 = self.multiscale_fuse3(self.upSample(x),encode_list[:3])
        x1, inter_feat = self.enhanceBlock1(x1, inter_feat)
        inter_feat.append(x1)
        x1 = self.proj1(x1)

        x2 = self.multiscale_fuse2(self.upSample(x1),encode_list[:3])
        x2, inter_feat = self.enhanceBlock2(x2, inter_feat)
        inter_feat.append(x2)
        x2 = self.proj2(x2)

        x3 = self.multiscale_fuse1(self.upSample(x2),encode_list[:3])
        x3, inter_feat = self.enhanceBlock3(x3, inter_feat)
        inter_feat.append(x3)
        x3 = self.proj3(x3)
        out = self.tail(x3)
        return out
