import torch.nn as nn
from .dice_loss import DiceLoss


class SegmentationLoss(nn.Module):

    def __init__(self, pos_weight):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, pred, target):

        loss = self.bce(pred, target) + self.dice(pred, target)

        return loss