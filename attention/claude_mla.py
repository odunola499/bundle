import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompressedCache:
    layer: int
    c_kv: Tensor  # Compressed KV representation
    k_r: Tensor  # Rotary position encoded keys


class RoPE:
    def __init__(self, dim: int, base: int = 10000):
        self.dim = dim
        self.base = base
        half_dim = dim // 2
        freqs = base ** (-torch.arange(0, half_dim).float() / half_dim)
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)
        self.register_buffer('freqs', freqs)

    def __call__(self, x: Tensor, pos: Tensor) -> Tensor:
        freqs = self.freqs.to(x.device)
        # pos should be [seq_len] and we want [seq_len, dim//2]
        theta = pos[:, None] * freqs[None, :]  # [seq_len, dim//2]
        sin, cos = torch.sin(theta), torch.cos(theta)

        # Reshape for broadcasting: [1, 1, seq_len, dim//2]
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]

        # Split x into even and odd indices
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        # Apply rotation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rotated


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, low_dim: int, rope_dim: int = None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.low_dim = low_dim
        self.head_dim = embed_dim // num_heads

        # RoPE dimension - can be different from head_dim
        self.rope_dim = rope_dim if rope_dim is not None else self.head_dim

        # Query projections
        self.W_DQ = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_UQ = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_QR = nn.Linear(embed_dim, self.rope_dim * num_heads, bias=False)

        # Key-Value projections
        self.W_DKV = nn.Linear(embed_dim, low_dim, bias=False)
        self.W_UK = nn.Linear(low_dim, embed_dim, bias=False)
        self.W_KR = nn.Linear(embed_dim, self.rope_dim, bias=False)  # Note: this operates on h_t directly
        self.W_UV = nn.Linear(low_dim, embed_dim, bias=False)

        # Output projection
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        # RoPE
        self.rope = RoPE(self.rope_dim)

        # Causal mask
        self.block_size = 2048
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            )
        )

    def forward(self, x: Tensor, cache: Optional[CompressedCache] = None) -> Tensor:
        batch_size, seq_len, embed_dim = x.shape

        # === Query Computation ===
        # c_Q_t = W_DQ @ h_t (Eq. 37)
        c_Q = self.W_DQ(x)  # [B, T, embed_dim]

        # q_C_t = W_UQ @ c_Q_t (Eq. 38)
        q_C = self.W_UQ(c_Q)  # [B, T, embed_dim]
        q_C = q_C.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q_C = q_C.transpose(1, 2)  # [B, num_heads, T, head_dim]

        # q_R_t = RoPE(W_QR @ c_Q_t) (Eq. 39)
        q_R_proj = self.W_QR(c_Q)  # [B, T, rope_dim * num_heads]
        q_R_proj = q_R_proj.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        q_R_proj = q_R_proj.transpose(1, 2)  # [B, num_heads, T, rope_dim]

        pos = torch.arange(seq_len, device=x.device)
        q_R = self.rope(q_R_proj, pos)  # [B, num_heads, T, rope_dim]

        # === Key-Value Computation ===
        # c_KV_t = W_DKV @ h_t (Eq. 41)
        c_KV = self.W_DKV(x)  # [B, T, low_dim]

        # k_C_t = W_UK @ c_KV_t (Eq. 42)
        k_C = self.W_UK(c_KV)  # [B, T, embed_dim]
        k_C = k_C.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_C = k_C.transpose(1, 2)  # [B, num_heads, T, head_dim]

        # k_R_t = RoPE(W_KR @ h_t) (Eq. 43)
        k_R_proj = self.W_KR(x)  # [B, T, rope_dim]
        k_R_proj = k_R_proj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, num_heads, T, rope_dim]
        k_R = self.rope(k_R_proj, pos)  # [B, num_heads, T, rope_dim]

        # v_C_t = W_UV @ c_KV_t (Eq. 45)
        v_C = self.W_UV(c_KV)  # [B, T, embed_dim]
        v_C = v_C.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_C = v_C.transpose(1, 2)  # [B, num_heads, T, head_dim]

        # === Attention Computation ===
        # Concatenate q_C and q_R for each head (Eq. 40)
        # Note: We need to handle the dimension mismatch if rope_dim != head_dim
        if self.rope_dim != self.head_dim:
            # Pad or truncate q_R to match head_dim
            if self.rope_dim < self.head_dim:
                pad_size = self.head_dim - self.rope_dim
                q_R_padded = F.pad(q_R, (0, pad_size))
            else:
                q_R_padded = q_R[:, :, :, :self.head_dim]
            q = torch.cat([q_C, q_R_padded], dim=-1)  # [B, num_heads, T, 2*head_dim]
        else:
            q = torch.cat([q_C, q_R], dim=-1)  # [B, num_heads, T, 2*head_dim]

        # Concatenate k_C and k_R for each head (Eq. 44)
        if self.rope_dim != self.head_dim:
            if self.rope_dim < self.head_dim:
                pad_size = self.head_dim - self.rope_dim
                k_R_padded = F.pad(k_R, (0, pad_size))
            else:
                k_R_padded = k_R[:, :, :, :self.head_dim]
            k = torch.cat([k_C, k_R_padded], dim=-1)  # [B, num_heads, T, 2*head_dim]
        else:
            k = torch.cat([k_C, k_R], dim=-1)  # [B, num_heads, T, 2*head_dim]

        # Compute attention scores
        scale = 1.0 / (self.head_dim + self.rope_dim) ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, T, T]

        # Apply causal mask
        if seq_len <= self.block_size:
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, T, T]

        # Apply attention to values (Eq. 46)
        out = torch.matmul(attn_weights, v_C)  # [B, num_heads, T, head_dim]

        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous()  # [B, T, num_heads, head_dim]
        out = out.view(batch_size, seq_len, embed_dim)  # [B, T, embed_dim]

        # Final output projection (Eq. 47)
        output = self.W_O(out)  # [B, T, embed_dim]

        return output

    def create_cache(self, layer_idx: int) -> CompressedCache:
        """Create an empty cache for this layer"""
        return CompressedCache(
            layer=layer_idx,
            c_kv=None,
            k_r=None
        )


# Example usage and testing
if __name__ == "__main__":
    # Test the implementation
    batch_size, seq_len, embed_dim = 2, 32, 512
    num_heads = 8
    low_dim = 128

    model = MultiHeadLatentAttention(embed_dim, num_heads, low_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compare with standard attention parameter count
    standard_attn_params = 4 * embed_dim * embed_dim  # Q, K, V, O projections
    mla_params = sum(p.numel() for p in model.parameters())
    print(f"Standard attention params: {standard_attn_params:,}")
    print(f"MLA params: {mla_params:,}")
    print(f"Parameter reduction: {(1 - mla_params / standard_attn_params) * 100:.1f}%")