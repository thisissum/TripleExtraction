import torch
from torch import nn

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, reduction="mean"):
        # assume inputs and targets have same shape
        # alpha * y * (1-p)^gamma * log(p) + (1-alpha) * (1-y) * p^gamma * log(1-p)
        loss = targets * torch.log(inputs) * torch.pow(1 - inputs, self.gamma) * self.alpha + \
               (1 - targets) * torch.log(1 - inputs) * torch.pow(inputs, self.gamma) * (1 - self.alpha)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction is None:
            return -loss
        return -loss