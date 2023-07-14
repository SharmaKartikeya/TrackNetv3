import torch.nn as nn

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
    
    def forward(self, output, target):
        return ((output-target)**2).sum()