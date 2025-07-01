import torch
from torch import nn, Tensor
from typing import Optional, Dict
from encoding import rotary_positional_encoding

class MultiLatentAttention(nn.Module):
    """
    MultiLatentAttention implements a memory-efficient multi-head attention
    by projecting keys and values into a lower-dimensional latent space and then
    reconstructing them. This reduces the size of the key-value cache. It can
    optionally apply RoPE (rotary positional embeddings) to Q and K.
    """
    def __init__(self, config, causal=True, rope=True):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim_model = config.dim_model
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v
        self.latent_dim = config.latent_dim
        self.is_causal = causal
        self.rope = rope

        # Linear for queries
        self.linear_q = nn.Linear(config.dim_model, config.dim_k * config.num_heads)

        # Linear projections to latent space
        self.latent_k_proj = nn.Linear(config.dim_model, self.latent_dim * config.num_heads)
        self.latent_v_proj = nn.Linear(config.dim_model, self.latent_dim * config.num_heads)

        # Reconstruction from latent space
        self.k_reconstruction = nn.Linear(self.latent_dim, config.dim_k, bias=False)
        self.v_reconstruction = nn.Linear(self.latent_dim, config.dim_v, bias=False)

        # Output linear
        self.linear_out = nn.Linear(config.dim_v * config.num_heads, config.dim_model)

    def forward_with_rope(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Tensor]] = None
    ):
        """
        Forward pass with rotary embeddings applied to Q and K.
        Maintains the same kv_cache structure as the main forward method.
        """
        batch_size, q_seq_len, _ = queries.shape
        _, k_seq_len, _ = keys.shape
        _, v_seq_len, _ = values.shape


        q = self.linear_q(queries)  # (B, Q, H*Dk)
        q = q.view(batch_size, q_seq_len, self.num_heads, self.dim_k)  # (B, Q, H, Dk)
        q = q.permute(0, 2, 1, 3)   # (B, H, Q, Dk)

        if kv_cache is not None:
            assert k_seq_len == 1, "If using KV cache, k_seq_len must be 1."
            if "k" in kv_cache and "v" in kv_cache:
                cached_k, cached_v = kv_cache["k"], kv_cache["v"]
                cached_latent_k, cached_latent_v = kv_cache["latent_k"], kv_cache["latent_v"]

                new_latent_k = self.latent_k_proj(keys)  # (B, 1, H*latent_dim)
                new_latent_v = self.latent_v_proj(values)

                new_latent_k = new_latent_k.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
                new_latent_v = new_latent_v.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
                new_latent_k = new_latent_k.permute(0, 2, 1, 3)  # (B, H, 1, latent_dim)
                new_latent_v = new_latent_v.permute(0, 2, 1, 3)

                new_k = self.k_reconstruction(new_latent_k)  # (B, H, 1, Dk)
                new_v = self.v_reconstruction(new_latent_v)  # (B, H, 1, Dv)

                kv_cache["k"] = torch.cat((cached_k, new_k), dim=2)  # (B, H, K+1, Dk)
                kv_cache["v"] = torch.cat((cached_v, new_v), dim=2)
                kv_cache["latent_k"] = torch.cat((cached_latent_k, new_latent_k), dim=2)
                kv_cache["latent_v"] = torch.cat((cached_latent_v, new_latent_v), dim=2)

            else:
                latent_k = self.latent_k_proj(keys)
                latent_v = self.latent_v_proj(values)
                latent_k = latent_k.view(batch_size, k_seq_len, self.num_heads, self.latent_dim).permute(0, 2, 1, 3)
                latent_v = latent_v.view(batch_size, v_seq_len, self.num_heads, self.latent_dim).permute(0, 2, 1, 3)

                processed_k = self.k_reconstruction(latent_k)
                processed_v = self.v_reconstruction(latent_v)

                kv_cache = {
                    "k": processed_k,
                    "v": processed_v,
                    "latent_k": latent_k,
                    "latent_v": latent_v
                }

            k, v = kv_cache["k"], kv_cache["v"]
        else:
            # No cache, just compute K, V fresh
            latent_k = self.latent_k_proj(keys)  # (B, K, H*latent_dim)
            latent_v = self.latent_v_proj(values)
            latent_k = latent_k.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
            latent_v = latent_v.view(batch_size, v_seq_len, self.num_heads, self.latent_dim)

            latent_k = latent_k.permute(0, 2, 1, 3)  # (B, H, K, latent_dim)
            latent_v = latent_v.permute(0, 2, 1, 3)
            k = self.k_reconstruction(latent_k)      # (B, H, K, Dk)
            v = self.v_reconstruction(latent_v)      # (B, H, K, Dv)

        # ---- Apply RoPE if enabled (to Q and K) ----
        if self.rope:
            rope_dim = self.dim_k // 2
            # Split Q
            q_left, q_right = torch.split(q, rope_dim, dim=-1)  # (B, H, Q, rope_dim) each
            q_right = rotary_positional_encoding(q_right)       # apply RoPE
            q = torch.cat((q_left, q_right), dim=-1)

            # Split K
            k_left, k_right = torch.split(k, rope_dim, dim=-1)
            k_right = rotary_positional_encoding(k_right)
            k = torch.cat((k_left, k_right), dim=-1)

        # ---- Attention scores ----
        # q: (B, H, Q, Dk), k: (B, H, K, Dk)
        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) / (self.dim_k ** 0.5)

        # ---- Apply mask ----
        if mask is not None:
            # (B, 1, 1, K)
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # ---- Causal mask ----
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(q_seq_len, k.shape[2], dtype=torch.bool, device=scores.device),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # ---- Softmax and compute weighted values ----
        attn_weights = torch.softmax(scores, dim=-1)
        # v: (B, H, K, Dv)
        out = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, v)

        # ---- Reshape and final linear ----
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, Q, H, Dv)
        out = out.view(batch_size, q_seq_len, self.dim_v * self.num_heads)
        out = self.linear_out(out)  # (B, Q, dim_model)

        return out, kv_cache

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Tensor]] = None
    ):
        """
        Standard forward pass without RoPE. Maintains and updates kv_cache if provided.
        """
        batch_size, q_seq_len, _ = queries.shape
        _, k_seq_len, _ = keys.shape
        _, v_seq_len, _ = values.shape

        # ---- Project queries ----
        q = self.linear_q(queries)  # (B, Q, H*Dk)
        q = q.view(batch_size, q_seq_len, self.num_heads, self.dim_k)
        q = q.permute(0, 2, 1, 3)   # (B, H, Q, Dk)

        # ---- Handle K, V with latent approach + cache ----
        if kv_cache is not None:
            assert k_seq_len == 1, "If using KV cache, k_seq_len must be 1."
            if "k" in kv_cache and "v" in kv_cache:
                cached_k, cached_v = kv_cache["k"], kv_cache["v"]
                cached_latent_k, cached_latent_v = kv_cache["latent_k"], kv_cache["latent_v"]

                new_latent_k = self.latent_k_proj(keys)
                new_latent_v = self.latent_v_proj(values)
                new_latent_k = new_latent_k.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
                new_latent_v = new_latent_v.view(batch_size, v_seq_len, self.num_heads, self.latent_dim)
                new_latent_k = new_latent_k.permute(0, 2, 1, 3)
                new_latent_v = new_latent_v.permute(0, 2, 1, 3)

                new_k = self.k_reconstruction(new_latent_k)
                new_v = self.v_reconstruction(new_latent_v)

                kv_cache["k"] = torch.cat((cached_k, new_k), dim=2)
                kv_cache["v"] = torch.cat((cached_v, new_v), dim=2)
                kv_cache["latent_k"] = torch.cat((cached_latent_k, new_latent_k), dim=2)
                kv_cache["latent_v"] = torch.cat((cached_latent_v, new_latent_v), dim=2)
            else:
                latent_k = self.latent_k_proj(keys)
                latent_v = self.latent_v_proj(values)
                latent_k = latent_k.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
                latent_v = latent_v.view(batch_size, v_seq_len, self.num_heads, self.latent_dim)
                latent_k = latent_k.permute(0, 2, 1, 3)
                latent_v = latent_v.permute(0, 2, 1, 3)

                processed_k = self.k_reconstruction(latent_k)
                processed_v = self.v_reconstruction(latent_v)

                kv_cache = {
                    "k": processed_k,
                    "v": processed_v,
                    "latent_k": latent_k,
                    "latent_v": latent_v
                }

            k, v = kv_cache["k"], kv_cache["v"]
        else:
            latent_k = self.latent_k_proj(keys)
            latent_v = self.latent_v_proj(values)
            latent_k = latent_k.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
            latent_v = latent_v.view(batch_size, v_seq_len, self.num_heads, self.latent_dim)
            latent_k = latent_k.permute(0, 2, 1, 3)
            latent_v = latent_v.permute(0, 2, 1, 3)
            k = self.k_reconstruction(latent_k)
            v = self.v_reconstruction(latent_v)

        # ---- Attention ----
        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) / (self.dim_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(q_seq_len, k.shape[2], dtype=torch.bool, device=scores.device), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, q_seq_len, self.dim_v * self.num_heads)
        out = self.linear_out(out)

        return out, kv_cache
