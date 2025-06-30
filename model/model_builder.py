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

# own files import
from .encoder import Encoder
from .decoder import Decoder
from .modules.urwkv import URWKV

# recursive network based on residual units
class LLENet(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.dim = dim

        self.encoder = Encoder(dim=self.dim) # 3 -> 32 -> 64 -> 128
        self.decoder = Decoder(dim=self.dim)  # 128 -> 64 -> 32 -> 3

        self.apply(self._init_weights)  # Correctly apply init_weights to all submodules

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        outer_shortcut = x
        inter_feat = []
        encode_list, inter_feat = self.encoder(x, inter_feat)

        x = encode_list[-1]
        x = self.decoder(x, encode_list, inter_feat)
        x=torch.add(x, outer_shortcut)

        return x


