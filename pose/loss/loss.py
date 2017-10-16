import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, size_average = True, weight=None):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss2d(weight, size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)
