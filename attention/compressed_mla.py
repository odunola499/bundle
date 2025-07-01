from torch import nn, Tensor
import torch
from dataclasses import dataclass
import math


@dataclass
class CompressedCache:
    layer: int
    c: Tensor
    k: Tensor


class Rope:
    def __init__(self, dim, base=10000):
        self.dim = dim
        self.base = base
        half_dim = dim // 2
        freqs = base ** (-torch.arange(0, half_dim).float() / half_dim)
        self.freqs = freqs

    def __call__(self, x, pos):
        freqs = self.freqs.to(x.device)
        theta = pos[:, None] * freqs[None, :]
        sin, cos = torch.sin(theta), torch.cos(theta)
        sin = sin[None, None, :, :]  # [1,1,T, half_dim]
        cos = cos[None, None, :, :]  # [1,1,T, half_dim]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiLatentAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, low_dim: int):
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

    def forward(self, x, past_cache=None, use_cache=False):
        batch_size, T, embed = x.shape

        if past_cache is not None and T == 1:
            past_ckv = past_cache.c
            past_kr_proj = past_cache.k
            past_T = past_ckv.shape[1]

            #[batch_size, 1, embed_dim]
            Cq = x @ self.Wdq
            Qc = Cq @ self.Wuq
            Qc = Qc.view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            Qr_proj = Cq @ self.Wqr
            Qr_proj = Qr_proj.view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            pos = torch.arange(past_T, past_T + 1, device=x.device)
            Qr = self.rope(Qr_proj, pos)

            q = torch.concat([Qc, Qr], dim=-1)

            new_ckv = x @ self.Wdkv  # [batch_size, 1, low_dim]
            new_kr_proj = x @ self.Wkr  # [batch_size, 1, embed_dim]

            Ckv = torch.cat([past_ckv, new_ckv], dim=1)  # [batch_size, past_T+1, low_dim]
            Kr_proj_flat = torch.cat([past_kr_proj, new_kr_proj], dim=1)  # [batch_size, past_T+1, embed_dim]

            Kc = Ckv @ self.Wuk
            Kc = Kc.view(batch_size, past_T + 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            Kr_proj = Kr_proj_flat.view(batch_size, past_T + 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            all_pos = torch.arange(0, past_T + 1, device=x.device)
            Kr = self.rope_k(Kr_proj, all_pos)

            k = torch.concat([Kc, Kr], dim=-1)

            v = Ckv @ self.Wuv  # [batch_size, past_T+1, embed_dim]
            v = v.view(batch_size, past_T + 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # update cache
            if use_cache:
                new_cache = CompressedCache(
                    layer=past_cache.layer,
                    c=Ckv,
                    k=Kr_proj_flat
                )
            else:
                new_cache = None

        else:
            Cq = x @ self.Wdq

            Qc = Cq @ self.Wuq
            Qc = Qc.view(batch_size, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            Qr_proj = Cq @ self.Wqr
            Qr_proj = Qr_proj.view(batch_size, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            pos = torch.arange(0, T, device=x.device)
            Qr = self.rope(Qr_proj, pos)

            q = torch.concat([Qc, Qr], dim=-1)

            Ckv = x @ self.Wdkv
            Kr_proj_flat = x @ self.Wkr

            Kc = Ckv @ self.Wuk
            Kc = Kc.view(batch_size, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            Kr_proj = Kr_proj_flat.view(batch_size, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            Kr = self.rope_k(Kr_proj, pos)

            k = torch.concat([Kc, Kr], dim=-1)

            v = Ckv @ self.Wuv
            v = v.view(batch_size, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            if use_cache:
                new_cache = CompressedCache(
                    layer=0, #current layer of this attention block
                    c=Ckv,
                    k=Kr_proj_flat
                )
            else:
                new_cache = None

        numerator = q @ k.transpose(-2, -1)
        divisor = math.sqrt(self.head_dim + self.head_dim)
        attn = numerator / divisor

        if past_cache is not None and T == 1:
            pass
        else:
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                                device=attn.device), diagonal=1)
            attn = attn.masked_fill(causal_mask, -float('inf'))

        attn_scores = torch.softmax(attn, -1)

        attn = attn_scores @ v
        output = attn.permute(0, 2, 1, 3).reshape(batch_size, -1, embed_dim)
        output = output @ self.Wo

        if use_cache:
            return output, new_cache
        return output


if __name__ == "__main__":
    num_heads = 8
    embed_dim = 512
    low_dim = 64

    compressed_c = torch.randn(2, num_heads,7, embed_dim)
    compressed_k = torch.randn(2,7,512)
    cache = CompressedCache(
        layer = 0,
        c = compressed_k,
        k = compressed_c
    )
    attention = MultiLatentAttention(num_heads, embed_dim, low_dim)

    tensor = torch.randn((2, 1,512))
    out = attention(tensor, past_cache = cache, use_cache = True)
    print(out)