import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_model = config.dim_model
        self.max_seq_len = config.max_seq_len

class AbsolutePositionalEncoding(PositionalEncoding):
    def __init__(self, config):
        super().__init__(config)
        pe = torch.zeros(self.max_seq_len, self.dim_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1)]

class RoPE(PositionalEncoding):
    def __init__(self, config):
        super().__init__(config)
        self.theta = config.rope_theta

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        freqs = torch.exp(torch.arange(0, head_dim, 2, device=x.device) * -(torch.log(torch.tensor(self.theta)) / head_dim))
        angles = positions * freqs
        
        sin_vals = torch.sin(angles).repeat(1, 2)
        cos_vals = torch.cos(angles).repeat(1, 2)
        
        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(batch_size, num_heads, seq_len, head_dim)
        x_enc = x * cos_vals + x_rot * sin_vals
        
        return x_enc
