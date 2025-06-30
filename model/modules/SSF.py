import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.conv = nn.Conv2d(in_channels * 4, 1, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Custom initialization for each convolution layer
        nn.init.kaiming_normal_(self.branch1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.branch3x3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.branch5x5.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.branch_pool.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(self.pool(x))

        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        outputs = self.conv(outputs)  
        return outputs

# State-aware Selective Fusion (SSF) module
class SSF(nn.Module):
    def __init__(self, num_feats, encode_channels, target_channels):
        super(SSF, self).__init__()
        self.num_feats = num_feats
        self.target_channels = target_channels

        # Alignment convolution layers for each encoder feature map
        self.align_convs = nn.ModuleList([
            nn.Conv2d(in_channels, target_channels, kernel_size=1) 
            for in_channels in encode_channels
        ])

        self.conv_fusion = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        self.inception = InceptionModule(num_feats, num_feats)
        self.final_conv = nn.Conv2d(target_channels * 2, target_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for conv in self.align_convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_fusion.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x, feat_list):
        target_height, target_width = x.size(2), x.size(3)
        feat_list.reverse()
        # Process each feature map in feat_list
        aligned_feats = []
        encoder_feat = torch.zeros_like(x)
        
        for i, feat in enumerate(feat_list):
            if feat.size(2) == target_height and feat.size(3) == target_width:
                encoder_feat = feat

            feat = torch.mean(feat, dim=1, keepdim=True)
            feat = F.interpolate(feat, size=(target_height, target_width), mode='bilinear', align_corners=False)
            aligned_feats.append(feat)

        # Stack the aligned feature maps along the channel dimension
        stacked_feats = torch.cat(aligned_feats, dim=1)

        # Fuse features along the N dimension
        fused_feat = self.conv_fusion(stacked_feats)

        # Apply the Inception module for multi-scale feature extraction
        inception_feat = self.inception(fused_feat)
        inception_feat = torch.sigmoid(inception_feat)

        guided_feat = inception_feat * encoder_feat

        output = self.final_conv(torch.cat([guided_feat, x], dim=1))

        return output

class NormLayer(nn.Module):
    def __init__(self, num_channels):
        super(NormLayer, self).__init__()
        
        # Learnable scaling and bias parameters
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
    def forward(self, x):
        # Apply normalization with learned scaling and bias
        return x * self.scale + self.bias
