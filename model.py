import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_c=16):
        super(UNet3D, self).__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_c, base_c*2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_c*2, base_c*4))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_c*4, base_c*8))
        
        # Decoder
        self.up1 = nn.ConvTranspose3d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_c*8, base_c*4)
        
        self.up2 = nn.ConvTranspose3d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_c*4, base_c*2)
        
        self.up3 = nn.ConvTranspose3d(base_c*2, base_c, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_c*2, base_c)
        
        self.outc = nn.Conv3d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # U-Net Skip Connections
        x = self.up1(x4)
        x = self.conv_up1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x1], dim=1))
        
        return self.outc(x)