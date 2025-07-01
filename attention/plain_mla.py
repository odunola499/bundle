import torch
import torch.nn as nn
import math

class RoPE:
    def __init__(self, dim, base=10000):
        self.dim = dim
        self.base = base
        half_dim = dim // 2
        freqs = base ** (-torch.arange(0, half_dim).float() / half_dim)
        self.freqs = freqs  # [half_dim]

    def __call__(self, x, pos):
        # x: (B, H, T, head_dim), pos: (T,)
        freqs = self.freqs.to(x.device)
        theta = pos[:, None] * freqs[None, :]  # [T, half_dim]
        sin, cos = torch.sin(theta), torch.cos(theta)
        sin = sin[None, None, :, :]  # [1, 1, T, half_dim]
        cos = cos[None, None, :, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RoPE(self.head_dim)

    def forward(self, x, cache=None, pos_offset=0):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(x): return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(reshape, (q, k, v))  # (B, H, T, head_dim)

        pos = torch.arange(pos_offset, pos_offset + T, device=x.device)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        if cache is not None:
            # cache: dict of {'k': (B,H,T_prev,D), 'v': (B,H,T_prev,D)}
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T_k)
        mask = torch.triu(torch.ones(T, k.shape[2], device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out), {'k': k.detach(), 'v': v.detach()}
