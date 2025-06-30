from typing import Sequence
import math, os

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import numbers
from .drop import DropPath


'''
# Global Average Pooling (GAP) as a Luminance Estimation
- The use of Global Average Pooling (GAP) on features, both on intermediate feature maps (inter_feat) and the input feature map (x), 
simulates a form of global luminance perception. The GAP extracts a summary of the overall brightness of an image, 
which is akin to how the human eye averages out brightness over large areas, 
especially in low-light conditions where local contrast is reduced.

- The GAP operation takes an average over spatial dimensions to create a single luminance descriptor for each channel, 
which can be considered as a global "luminance estimation."
'''


class LuminanceAdaptiveNorm(nn.Module):
    def __init__(self, dim, channel_first=True, seed=42):
        super().__init__()

        torch.manual_seed(seed)

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling to [B, C', 1, 1]
        
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, dim),  
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.mlp[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.mlp[2].weight)

    def _gap_and_pad_features(self, inter_feat, x_):
        """Applies GAP to input features and pads channels to match the maximum."""
        gap_feats = [self.gap(feat) for feat in inter_feat]  # [B, C', 1, 1]
        gap_x = self.gap(x_)
        gap_feats.append(gap_x)

        # Find maximum channels in inter_feat and apply GAP with padding
        max_channels = max([feat.shape[1] for feat in gap_feats])

        # Zero-pad features to match the maximum number of channels
        padded_feats = []
        for feat in gap_feats:
            B, C_gap, _, _ = feat.shape
            if C_gap < max_channels:
                padding = torch.zeros(B, max_channels - C_gap, 1, 1, device=feat.device)
                feat = torch.cat([feat, padding], dim=1)  # Concatenate along the channel dimension
            padded_feats.append(feat)

        return torch.stack(padded_feats, dim=0).squeeze(-1).squeeze(-1)  # [num_feats, B, max_C]

    def _apply_convolutions(self, stacked_feats, num_feats, x_device):
        """Applies convolution operations with different kernel sizes."""
        conv_results = []

        for kernel_size in [1, 3, 5]:
            conv_layer = nn.Conv2d(1, 1, kernel_size=(num_feats, kernel_size), padding=0, bias=False).to(x_device)
            conv_out = conv_layer(stacked_feats).squeeze(2).squeeze(1)  # Squeeze unnecessary dimensions
            conv_results.append(conv_out)

        return conv_results

    def _apply_linear_layers(self, conv_results, C, x_device):
        """Apply Linear layers to match the feature channels with C."""
        linear_results = []

        for conv_out in conv_results:
            linear_layer = nn.Linear(conv_out.shape[-1], C).to(x_device)
            linear_results.append(linear_layer(conv_out))

        return torch.cat(linear_results, dim=-1)  

    def forward(self, x, inter_feat, patch_resolution):
        if x.dim() == 4:
            B, _, N, C = x.shape
            x = x.view(B, N, C)
        else:
            B, N, C = x.shape        
        H_x, W_x = patch_resolution

        # Reshape and permute x to match the input format for GAP
        x_reshaped = x.view(B, H_x, W_x, C).permute(0, 3, 1, 2)

        stacked_feats = self._gap_and_pad_features(inter_feat, x_reshaped)
        stacked_feats = stacked_feats.permute(1, 0, 2).unsqueeze(1)  # [B, 1, num_feats, max_C]

        # Apply convolutions with different kernel sizes (1, 3, 5)
        conv_results = self._apply_convolutions(stacked_feats, len(inter_feat) + 1, x.device)

        # Apply linear layers to match the feature dimension C
        concatenated_features = self._apply_linear_layers(conv_results, C, x.device)

        conv_out = torch.tanh(self.mlp(concatenated_features))  # [B, C]

        # Adjust alpha using the learned conv_out features
        adjusted_alpha = self.alpha + conv_out.view(B, 1, C)

        # Normalize x and apply the adjustments
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mu) / (sigma + 1e-3)

        # Apply color transform and adjustments based on channel_first flag
        if self.channel_first:
            x_transformed = torch.tensordot(x_normalized, self.color, dims=[[-1], [-1]])
            x_out = x_transformed * adjusted_alpha + self.beta
        else:
            x_out = x_normalized * adjusted_alpha + self.beta
            x_out = torch.tensordot(x_out, self.color, dims=[[-1], [-1]])

        return x_out.view(B, N, C)



