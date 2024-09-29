import torch.nn as nn


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
