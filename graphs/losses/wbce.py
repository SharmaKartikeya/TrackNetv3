import torch
import torch.nn as nn

class WBCE(nn.Module):
    def __init__(self, pos_factor: float):
        super(WBCE, self).__init__()
        pos_weight = torch.tensor(pos_factor, dtype=torch.float64)
        self.loss = nn.BCEWithLogitsLoss(pos_weight)

    def forward(self, output, target):
        return self.loss(output, target)