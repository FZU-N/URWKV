import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.urwkv import URWKV

class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.patch_size = 4
        self.head = nn.Conv2d(3, self.dim, kernel_size=3, stride=1, padding=1, bias=False)

        # stage1 
        self.enhanceBlock1 = URWKV(patch_size=3, in_channels=self.dim, embed_dims=self.dim, depth=3)
        self.proj1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # stage2
        self.enhanceBlock2 = URWKV(patch_size=3, in_channels=self.dim, embed_dims=self.dim, depth=3)
        self.proj2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # stage3
        self.enhanceBlock3 = URWKV(patch_size=3, in_channels=self.dim*2, embed_dims=self.dim*2, depth=3)
        self.proj3 = nn.Sequential(
            nn.Conv2d(self.dim*2, self.dim*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x, inter_feat):
        B, C, H, W = x.shape
        x = self.head(x)
        inter_feat.append(x)  # [1, 32, 256, 256]

        # stage1
        x1, inter_feat = self.enhanceBlock1(x, inter_feat)
        inter_feat.append(x1)  # [1, 32, 256, 256], [1, 32, 256, 256]
        x1 = self.proj1(x1)  # C, H, W  
        x1_out = F.interpolate(x1, scale_factor=0.5, mode='bilinear')  # down 1/2
        
        # stage2
        x1_out, inter_feat = self.enhanceBlock2(x1_out, inter_feat)
        inter_feat.append(x1_out) # [1, 32, 256, 256], [1, 32, 256, 256], [1, 32, 128, 128]
        x2 = self.proj2(x1_out)  # 2C, H/2, W/2
        x2_out = F.interpolate(x2, scale_factor=0.5, mode='bilinear')  # down 1/4
     
        # stage3
        x2_out, inter_feat = self.enhanceBlock3(x2_out, inter_feat)
        inter_feat.append(x2_out) # [1, 32, 256, 256], [1, 32, 256, 256], [1, 32, 128, 128], [1, 64, 64, 64]
        x3 = self.proj3(x2_out)  # 4C, H/4, W/4
        x3_out = F.interpolate(x3, scale_factor=0.5, mode='bilinear')  # down 1/8, 4C, H/8, W/8
        
        feat_list = [x1, x2, x3, x3_out]
        return feat_list, inter_feat