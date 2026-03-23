import torch 
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConV,Down,Up,OutConv

class UnetWithClassifier(nn.Module):
    def __init__(self,n_channels=1,n_seg_classes=1,n_clf_classes=4,bilinear=False):
        super().__init__()
        self.inp = DoubleConV(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512,1024//factor)
        self.up1 = Up(1024,512 // factor,bilinear)
        self.up2 = Up(512,256 // factor,bilinear)
        self.up3 = Up(256, 128//factor,bilinear)
        self.up4 = Up(128,64,bilinear)
        self.outp = OutConv(64,n_seg_classes)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),
            nn.Linear(1024//factor,512),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(512,n_clf_classes)
        )

    def forward(self,x):
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        clf = self.classifier(x5) # determine tumor category
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1) # generating mask
        return self.outp(x),clf