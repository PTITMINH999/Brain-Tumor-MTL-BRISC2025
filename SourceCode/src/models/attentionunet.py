import torch 
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConV,Down,AttentionUp,OutConv

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inp = DoubleConV(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = AttentionUp(1024, 512)
        self.up2 = AttentionUp(512, 256)
        self.up3 = AttentionUp(256, 128)
        self.up4 = AttentionUp(128, 64)
        self.outp = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inp(x) 
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) 
        x5 = self.down4(x4)
        x = self.up1(x5, x4) 
        x = self.up2(x, x3) 
        x = self.up3(x, x2) 
        x = self.up4(x, x1)
        return self.outp(x)