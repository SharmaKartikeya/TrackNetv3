import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self, pos_factor):
        super(MyLoss, self).__init__()
        self.pos_factor = pos_factor

    def forward(self, output, target):
        diff = output - target
        diff[diff<0] *= self.pos_factor
        return (diff**2).mean()
