from torch import nn, Tensor
import torch
from dataclasses import dataclass
import math


@dataclass
class CompressedCache:
    layer:int
    c: Tensor
    k: Tensor


class Rope:
    def __init__(self, dim, base = 10000):
        self.dim = dim
        self.base = base
        half_dim = dim // 2
        freqs = base ** (-torch.arange(0, half_dim).float() / half_dim)
        self.freqs = freqs

    def __call__(self, x, pos):
        freqs = self.freqs.to(x.device)
        theta = pos[:, None] * freqs[None, :]
        sin, cos = torch.sin(theta), torch.cos(theta)
        sin = sin[None, None, :, :] #[1,1,T, half_dim]
        cos = cos[None, None, :, :]#[1,1,T, half_dim]
        x1 = x[...,0::2]
        x2 = x[...,1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiLatentAttention(nn.Module):
    def __init__(self, num_heads:int, embed_dim:int , low_dim:int):
        super().__init__()

        self.block_size = 1024
        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads

        self.Wdq = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wuq = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wqr = nn.Parameter(torch.randn(embed_dim, self.head_dim * self.num_heads))

        self.rope = Rope(self.head_dim)
        self.rope_k = Rope(self.head_dim)

        self.Wdkv = nn.Parameter(torch.randn(embed_dim, low_dim))
        self.Wuk = nn.Parameter(torch.randn(low_dim, embed_dim))
        self.Wkr = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wuv = nn.Parameter(torch.randn(low_dim, embed_dim))

        self.Wo = nn.Parameter(torch.randn(embed_dim, embed_dim))


        self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                             .view(1, 1, self.block_size, self.block_size))

    def forward(self, x):
        batch_size, T, embed = x.shape

        Cq = x @ self.Wdq #eqn 37 [batch_size,t, embed_dim ]

        # queries without rope
        Qc = Cq @ self.Wuq #eqn38 [batch_size,t, embed_dim ]
        Qc = Qc.view(batch_size, T, self.num_heads,self.head_dim).permute(0,2,1,3) #eqn38 expand[batch_size, num_head, T, head_dim]

        # queries with rope
        Qr_proj = Cq @ self.Wqr #eqn 39 [batch_size, T, embed_dim]

        Qr_proj = Qr_proj.view(batch_size, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        pos = torch.arange(0, T, device=x.device)

        Qr = self.rope(Qr_proj, pos) #eqn 39 expand [batch_size, num_head, T, head_dim]

        #queries! [batch_size,num_heads, T, head_dim *2]
        q = torch.concat([Qc, Qr], dim = -1) #eqn 40 [batch_size,num_heads, T, head_dim *2]

        #batch_size, T , low dim
        Ckv = x @ self.Wdkv #eqn 41 [batch_size, T, low_dim] #cached during inference

        #get keys without rope from Ckv
        Kc =  Ckv @ self.Wuk #eqn 42 [batch_size, T, embed_dim]
        Kc = Kc.view(batch_size, T, self.num_heads, self.head_dim).permute(0,2,1,3) #eqn 42 expanded [batch_size, num_head, T, head_dim]

        #keys with rope from Kc
        Kr_proj = x @ self.Wkr #[batch_size, T, embed_dim] #cached during inference
        Kr_proj = Kr_proj.view(batch_size, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        Kr = self.rope_k(Kr_proj, pos) #[batch_size, num_heads, T, head_dim] eqn 43

        #keys!!
        k = torch.concat([Kc, Kr], dim = -1) #en 44  [batch_size,num_heads, T, head_dim *2]


        #values!!
        v = Ckv @ self.Wuv #[batch_size, T, embed_dim] eqn 45
        v = v.view(batch_size, T, self.num_heads, self.head_dim).permute(0,2,1,3)

        #attn, eqn 45 - 47
        numerator = q @ k.transpose(-2,-1)
        divisor = math.sqrt(self.head_dim + self.head_dim)
        attn = numerator / divisor

        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                            device=attn.device), diagonal=1)
        attn = attn.masked_fill(causal_mask, -float('inf'))
        attn_scores = torch.softmax(attn, -1)

        attn = attn_scores @ v
        output = attn.permute(0, 2, 1, 3).reshape(batch_size, T, embed_dim)
        output = output @ self.Wo

        return output


if __name__ == "__main__":
    num_heads = 8
    embed_dim = 512
    low_dim = 64
    attention = MultiLatentAttention(num_heads, embed_dim, low_dim)

    tensor = torch.randn((2, 8,512))
    out = attention(tensor)