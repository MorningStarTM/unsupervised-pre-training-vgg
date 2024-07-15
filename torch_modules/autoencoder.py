import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x128

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16

            # Conv Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
        )

    def forward(self, x):
        x = self.encoder(x)
        return x




class VGGDecoder(nn.Module):
    def __init__(self):
        super(VGGDecoder, self).__init__()
        self.decoder = nn.Sequential(
            # Deconv Block 1
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),  # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Deconv Block 2
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Deconv Block 3
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),  # 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Deconv Block 4
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Deconv Block 5
            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2),  # 256x256
            nn.Sigmoid()  # Use sigmoid to ensure the output is between 0 and 1
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class VGGAutoencoder(nn.Module):
    def __init__(self):
        super(VGGAutoencoder, self).__init__()
        self.encoder = VGGEncoder()
        self.decoder = VGGDecoder()

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x