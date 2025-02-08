import torch
from torch import nn
from torch import Tensor

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_length = config.max_length
        self.dim_model = config.dim_model
        self.pe = nn.Linear(self.max_length, self.dim_model)

    def forward(self, x:Tensor):
        output = x + self.pe(x)
        return output

    def apply_pos(self, x:Tensor):
        return self.forward(x)