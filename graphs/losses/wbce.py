import torch
import torch.nn as nn

class WBCE(nn.Module):
    def __init__(self, pos_factor: float):
        super(WBCE, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_factor))

    def forward(self, output, target):
        return self.loss(output, target)