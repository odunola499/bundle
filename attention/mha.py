import torch
from torch import nn
from torch import Tensor
from typing import Optional, Dict
from encoding import rotary_positional_encoding

class MultiHeadAttention(nn.Module):
    def __init__(self, config, causal=True, rope=True):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim_model = config.dim_model
        #TODO: ALlow user to choose dim_q
        self.dim_q = config.dim_model // config.num_heads
        self.dim_k = config.dim_model // config.num_heads
        self.dim_v = config.dim_model // config.num_heads
        self.rope = rope  # whether to use RoPE encoding
        self.is_causal = causal  # whether the attention is causal

        self.linear_q = nn.Linear(config.dim_model, self.dim_q * config.num_heads)
        self.linear_k = nn.Linear(config.dim_model, self.dim_k * config.num_heads)
        self.linear_v = nn.Linear(config.dim_model, self.dim_v * config.num_heads)
        self.linear_out = nn.Linear(config.dim_v * config.num_heads, config.dim_model)

        assert self.dim_v == self.dim_k , "dimensions of values and keys should be the same"
    def forward(self, queries: Tensor,
                keys: Tensor,
                values: Tensor,
                mask: Optional[Tensor] = None,  # clarify expected shape (e.g., [batch, k_seq_len] for key mask)
                kv_cache: Optional[Dict[str, Tensor]] = None):
        batch_size, q_seq_len, _ = queries.shape
        _, k_seq_len, _ = keys.shape
        _, v_seq_len, _ = values.shape

        # Compute queries
        q = self.linear_q(queries)
        q = q.view(batch_size, q_seq_len, self.num_heads, self.dim_k).permute(0, 2, 1, 3)

        # Compute keys and values (with optional KV cache)
        if kv_cache is not None:
            assert k_seq_len == 1, "Since KV Cache is enabled, k_seq_len must be 1"
            if "k" in kv_cache and "v" in kv_cache:
                cached_k, cached_v = kv_cache["k"], kv_cache["v"]
                new_k = self.linear_k(keys)
                new_v = self.linear_v(values)
                new_k = new_k.view(batch_size, q_seq_len, self.num_heads, self.dim_k).permute(0, 2, 1, 3)
                new_v = new_v.view(batch_size, k_seq_len, self.num_heads, self.dim_v).permute(0, 2, 1, 3)
                kv_cache["k"] = torch.cat((cached_k, new_k), dim=2)
                kv_cache["v"] = torch.cat((cached_v, new_v), dim=2)
            else:
                k = self.linear_k(keys)
                v = self.linear_v(values)
                k = k.view(batch_size, k_seq_len, self.num_heads, self.dim_k).permute(0, 2, 1, 3)
                v = v.view(batch_size, v_seq_len, self.num_heads, self.dim_v).permute(0, 2, 1, 3)
                kv_cache = {"k": k, "v": v}
            k, v = kv_cache['k'], kv_cache['v']
        else:
            k = self.linear_k(keys)
            v = self.linear_v(values)
            k = k.view(batch_size, k_seq_len, self.num_heads, self.dim_k).permute(0, 2, 1, 3)
            v = v.view(batch_size, v_seq_len, self.num_heads, self.dim_v).permute(0, 2, 1, 3)

        # Apply rotary positional encoding if enabled
        if self.rope:
            q = rotary_positional_encoding(q)
            k = rotary_positional_encoding(k)

        # Compute scaled dot-product attention scores
        scores = torch.einsum("bnqd, bnkd -> bnqk", q, k) / (self.dim_k ** 0.5)

        # Apply mask if provided.
        # If mask is for keys (shape [batch, k_seq_len]):
        #   mask = mask.unsqueeze(1).unsqueeze(2)
        # If mask is for queries (shape [batch, q_seq_len]):
        #   mask = mask.unsqueeze(1).unsqueeze(3)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # adjust as needed
            scores = scores.masked_fill(mask, -float('inf'))

        # Apply causal mask if necessary
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_seq_len, k_seq_len, dtype=torch.bool,
                                                  device=scores.device), diagonal=1)
            scores = scores.masked_fill(causal_mask, -float('inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        # Compute attention output using corrected einsum
        attn_output = torch.einsum("bnqk, bnkd -> bnqd", attn_weights, v)
        # Reshape back to (batch_size, q_seq_len, num_heads * dim_v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, q_seq_len, self.num_heads * self.dim_v)
        y = self.linear_out(attn_output)
        return y, kv_cache
