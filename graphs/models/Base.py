import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt

    def forward(self, x):
        raise NotImplementedError


    def save(self, path, whole_model=False):
        raise NotImplementedError


    def load(self, path, device='cpu'):
        raise NotImplementedError