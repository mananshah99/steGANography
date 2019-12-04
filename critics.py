# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision.models import densenet, resnet
import torch.nn.functional as F

class BasicCritic(nn.Module):
    """
    The BasicCritic module takes an image and predicts whether it is a cover
    image or a steganographic image (N, 1).

    Input: (N, 3, H, W)
    Output: (N, 1)
    """

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3
        )

    def _build_models(self):
        return nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, 1)
        )

    def __init__(self, hidden_size):
        super().__init__()
        self.version = '1'
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self._models = self.layers
            self.version = '1'

    def forward(self, x):
        x = self._models(x)
        x = torch.mean(x.view(x.size(0), -1), dim=1)
        return x

class DenseCritic(BasicCritic):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        
        self._models = densenet.densenet121(pretrained=True)
        self._models.train()

    def forward(self, x):
        features = self._models.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = torch.mean(out.view(out.size(0), -1), dim=1)
        return out 

class ResidualCritic(BasicCritic):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        
        self._models = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=2)
        self._models.train()

    def forward(self, x):
        x = self._models.conv1(x)
        x = self._models.bn1(x)
        x = self._models.relu(x)
        x = self._models.maxpool(x)

        x = self._models.layer1(x)
        x = self._models.layer2(x)
        x = self._models.layer3(x)
        x = self._models.layer4(x)

        x = self._models.avgpool(x)
        x = torch.mean(x.view(x.size(0), -1), dim=1)
        return x[0]
