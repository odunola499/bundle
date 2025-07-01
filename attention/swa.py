import torch
from torch import nn
from torch import Tensor
from typing import Optional, Dict

class SlidingWindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim_model = config.dim_model
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v
        self.window_size = config.window_size

