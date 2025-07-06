from torch import nn
import torch
from attention import MultiHeadAttention
from feedforward import FeedForward

from dataclasses import dataclass

@dataclass
class ModelConfig:
    dim_ff:int = 2048
    dim_k:int = 64
    dim_v:int = 64
    num_heads:int = 8
    dim_model = 512
    is_causal = True

    dropout:float = 0.1
    max_seq_len = 512


if __name__ == "__main__":
    config = ModelConfig()
    attention = MultiHeadAttention(config)
    tensor = torch.randn((8, 16, 512))
    print(tensor.shape)
    output, _ = attention(tensor, tensor, tensor)
    print(output.shape)


