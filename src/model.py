import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.enc1 = self.double_conv(n_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.double_conv(128 + 64, 64)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x2 = self.enc2(x2)
        x = self.up(x2)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        return self.final(x)