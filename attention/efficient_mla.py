from torch import nn, Tensor
import torch
from dataclasses import dataclass
import math

@dataclass
class MLACache:
    layer:int
    ckv: Tensor
    kr_proj: Tensor

class EfficientMLA(nn.Module):
    def __init__(self, num_heads:int, embed_dim:int, low_dim:int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.low_dim = low_dim

        self.Wdq = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wuq = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wqr = nn.Parameter(torch.randn(embed_dim, self.head_dim))

        self.Wdkv = nn.Parameter(torch.randn(embed_dim, low_dim))
        self.Wkr = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # first fused matrix
        self.fused_upsample = nn.Parameter(torch.randn(low_dim, embed_dim * 2))
        # second fused matrix
        self.Wo = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x:Tensor, cache: MLACache):
        batch_size, T, _ = x.shape

        current_ckv =  x @ self.Wdkv
        current_kr_proj = x @ self.Wkr

        current_upsampled = current_ckv @ self.fused_upsample

        if cache.ckv is None:
            #initialise the cache
            cache.ckv = current_ckv
            cache.kr_proj = current_kr_proj
            cache.upsampled_kv = current_upsampled
            seq_len = 1

        else:
            cache.ckv = torch.cat([cache.ckv, current_ckv], dim=1)
            cache.kr_proj = torch.cat([cache.kr_proj, current_kr_proj], dim=1)
            cache.upsampled_kv = torch.cat([cache.upsampled_kv, current_upsampled], dim=1)
            seq_len = cache.ckv.shape[1]

        Cq = x @ self.Wdq

        #queries, nothing much has changed
        Qc = Cq @ self.Wuq
        Qr_proj = Cq @ self.Wqr

        Qc = Qc.view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Qr_proj = Qr_proj.view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        pos = torch.arange(0, seq_len, device = x.device)
        Qr = self.rope(Qr_proj,pos)

        q = torch.cat([Qc, Qr], dim = -1)

        k_upsampled, v = torch.split(cache.upsampled_kv, self.embed_dim, dim = -1)

        Kc = k_upsampled.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)

        Kr_proj_reshaped = cache.kr_proj.unsqueeze(2).repeat(1, 1, self.num_heads, 1).permute(0, 2, 1, 3)
        positions = torch.arange(seq_len, device=x.device)
        Kr = self.rope_k(Kr_proj_reshaped, positions)

        #keys!!
        k = torch.cat([Kc, Kr], dim=-1)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = attn_scores / math.sqrt(self.head_dim * 2)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v

        output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, 1, self.embed_dim)
        output = output @ self.Wo

        return output, cache