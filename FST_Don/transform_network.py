import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class TransformNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        # Initial convs (Downsampling)
        self.conv_down = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        ).to(self.device)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        ).to(self.device)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        ).to(self.device)

        # Final conv
        self.final = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4).to(
            self.device
        )

    def forward(self, x):
        x = self.conv_down(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        x = self.final(x)
        return torch.tanh(x)  # Output in [-1, 1]
