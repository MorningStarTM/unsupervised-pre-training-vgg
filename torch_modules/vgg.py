import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, kernel_size=6, stride=1, padding=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def upconv_block(self, in_channels, out_channels, kernel_size=6, stride=1, padding=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='nearest')
    )


class VGGEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super(VGGEncoder, self).__init__()
        self.block1 = conv_block(input_channels, 16)
        self.block2 = conv_block(16, 32)
        self.block3 = conv_block(32, 64)
        self.block4 = conv_block(64, 128)
        self.block5 = conv_block(128, 256)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
    

class VGGDecoder(nn.Module):
    def __init__(self):
        super(VGGDecoder, self).__init__()
        self.upconv1 = self.upconv_block(256, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv3 = self.upconv_block(128, 64)
        self.upconv4 = self.upconv_block(64, 32)
        self.upconv5 = self.upconv_block(32, 16)
        self.final = nn.ConvTranspose2d(16, 3, kernel_size=6, stride=1, padding=2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.final(x)
        x = self.activation(x)
        return x

