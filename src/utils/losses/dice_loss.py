import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets, smooth: int=1):
        num = targets.size(0)

        logits = torch.sigmoid(logits).view(num, -1)
        targets = targets.view(num, -1)

        intersection = 2. * (logits * targets).sum() + smooth
        union = logits.sum() + targets.sum() + smooth

        dice_coeff = 1 - (intersection / union)

        bce_loss = nn.BCELoss()(logits, targets)

        return bce_loss + dice_coeff
