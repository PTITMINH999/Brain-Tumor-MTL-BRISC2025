import torch 
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConV,Down,Up,OutConv

class UnetClassifier(nn.Module):
    def __init__(self,n_channels=1,n_classes=4):
        super().__init__()
        self.inp = DoubleConV(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,512)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        x = self.inp(x); 
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x) 
        x = self.down4(x)
        return self.classifier(x)
