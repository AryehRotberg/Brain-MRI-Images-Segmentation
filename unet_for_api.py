from typing import Tuple

import torch
import torch.nn as nn

import torchvision.transforms.functional as TF


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.batchnorm_1 = nn.BatchNorm2d(out_channels)
        self.batchnorm_2 = nn.BatchNorm2d(out_channels)
        
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        x = self.relu_1(x)
        
        x = self.conv_2(x)
        x = self.batchnorm_2(x)
        
        return self.relu_2(x)


class UNET(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 block_sizes: Tuple[int]=(64, 128, 256, 512)):
        
        super(UNET, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for block_size in block_sizes:
            self.encoder.append(DoubleConvBlock(in_channels, block_size))
            in_channels = block_size
            
        for block_size in block_sizes[::-1]:
            self.decoder.append(nn.ConvTranspose2d(2 * block_size, block_size, kernel_size=2, stride=2))
            self.decoder.append(DoubleConvBlock(2 * block_size, block_size))
        
        self.bottleneck = DoubleConvBlock(block_sizes[-1], 2 * block_sizes[-1])
        self.output_conv = nn.Conv2d(block_sizes[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        concatenations = []
        
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            concatenations.append(x)
            x = self.max_pool(x)
            
        x = self.bottleneck(x)
        concatenations = concatenations[::-1]
        
        for _ in range(0, len(self.decoder), 2):
            x = self.decoder[_](x)
            encoder_layer = concatenations[_ // 2]

            if x.shape != encoder_layer.shape:
                x = TF.resize(x, size=encoder_layer.shape[2:])
            
            concat_layer = torch.cat((encoder_layer, x), dim=1)
            
            x = self.decoder[_ + 1](concat_layer)
            
        return self.output_conv(x)
