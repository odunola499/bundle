import torch
from torch import nn, Tensor
from typing import Optional, Dict
from attention import AttentionConfig


class Attention(nn.Module):
    def __init__(self, config:AttentionConfig):
        super().__init__()
        self.num_heads: int = config.num_heads
        self.dim_model: int = config.dim_model
        self.num_group_heads: int = config.num_group_heads if hasattr(config, 'num_group_heads') else self.num_heads
        self.dim_k: int = config.dim_k
        self.dim_v: int = config.dim_v
        self.is_causal: bool = config.is_causal


class MultiHeadAttention(Attention):
    def __init__(self, config):
        super().__init__(config)
        self.linear_q = nn.Linear(self.dim_model, self.dim_k * self.num_heads)
        self.linear_k = nn.Linear(self.dim_model, self.dim_k * self.num_heads)
        self.linear_v = nn.Linear(self.dim_model, self.dim_v * self.num_heads)
        self.linear_out = nn.Linear(self.dim_v * self.num_heads, self.dim_model)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Optional[Tensor] = None,
                kv_cache: Optional[Dict[str, Tensor]] = None):
        batch_size, q_seq_len, _ = queries.shape
        _, k_seq_len, _ = keys.shape

        q = self.linear_q(queries)
        q = q.view(batch_size, q_seq_len, self.num_heads, self.dim_k)
        q = q.permute(0, 2, 1, 3)

        if kv_cache is not None:
            if "k" in kv_cache and "v" in kv_cache:
                k_new = self.linear_k(keys)
                k_new = k_new.view(batch_size, k_seq_len, self.num_heads, self.dim_k)
                k_new = k_new.permute(0, 2, 1, 3)
                k = torch.cat([kv_cache["k"], k_new], dim=2)

                v_new = self.linear_v(values)
                v_new = v_new.view(batch_size, k_seq_len, self.num_heads, self.dim_v)
                v_new = v_new.permute(0, 2, 1, 3)
                v = torch.cat([kv_cache["v"], v_new], dim=2)
            else:
                k = self.linear_k(keys)
                k = k.view(batch_size, k_seq_len, self.num_heads, self.dim_k)
                k = k.permute(0, 2, 1, 3)

                v = self.linear_v(values)
                v = v.view(batch_size, k_seq_len, self.num_heads, self.dim_v)
                v = v.permute(0, 2, 1, 3)
            kv_cache["k"] = k
            kv_cache["v"] = v
        else:
            k = self.linear_k(keys)
            k = k.view(batch_size, k_seq_len, self.num_heads, self.dim_k)
            k = k.permute(0, 2, 1, 3)

            v = self.linear_v(values)
            v = v.view(batch_size, k_seq_len, self.num_heads, self.dim_v)
            v = v.permute(0, 2, 1, 3)

        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        scores = scores / (self.dim_k ** 0.5)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, float('-inf'))

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_seq_len, k.shape[2], dtype=torch.bool, device=scores.device),
                                     diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, v)
        out = out.permute(0, 2, 1, 3)
        out = out.contiguous()
        out = out.view(batch_size, q_seq_len, self.dim_v * self.num_heads)
        out = self.linear_out(out)
        return out, kv_cache


class MultiQueryAttention(Attention):
    def __init__(self, config):
        super().__init__(config)
        self.linear_q = nn.Linear(self.dim_model, self.dim_k * self.num_heads)
        self.linear_k = nn.Linear(self.dim_model, self.dim_k)
        self.linear_v = nn.Linear(self.dim_model, self.dim_v)
        self.linear_out = nn.Linear(self.dim_v * self.num_heads, self.dim_model)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Optional[Tensor] = None,
                kv_cache: Optional[Dict[str, Tensor]] = None):
        batch_size, q_seq_len, _ = queries.shape
        _, k_seq_len, _ = keys.shape

        q = self.linear_q(queries)
        q = q.view(batch_size, q_seq_len, self.num_heads, self.dim_k)
        q = q.permute(0, 2, 1, 3)

        if kv_cache is not None:
            if "k" in kv_cache and "v" in kv_cache:
                k_new = self.linear_k(keys)
                k_new = k_new.view(batch_size, k_seq_len, 1, self.dim_k)
                k_new = k_new.permute(0, 2, 1, 3)
                k = torch.cat([kv_cache["k"], k_new], dim=2)

                v_new = self.linear_v(values)
                v_new = v_new.view(batch_size, k_seq_len, 1, self.dim_v)
                v_new = v_new.permute(0, 2, 1, 3)
                v = torch.cat([kv_cache["v"], v_new], dim=2)
            else:
                k = self.linear_k(keys)
                k = k.view(batch_size, k_seq_len, 1, self.dim_k)
                k = k.permute(0, 2, 1, 3)

                v = self.linear_v(values)
                v = v.view(batch_size, k_seq_len, 1, self.dim_v)
                v = v.permute(0, 2, 1, 3)
            kv_cache["k"] = k
            kv_cache["v"] = v
        else:
            k = self.linear_k(keys)
            k = k.view(batch_size, k_seq_len, 1, self.dim_k)
            k = k.permute(0, 2, 1, 3)

            v = self.linear_v(values)
            v = v.view(batch_size, k_seq_len, 1, self.dim_v)
            v = v.permute(0, 2, 1, 3)

        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        scores = scores / (self.dim_k ** 0.5)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, float('-inf'))

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_seq_len, k.shape[2], dtype=torch.bool, device=scores.device),
                                     diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, v)
        out = out.permute(0, 2, 1, 3)
        out = out.contiguous()
        out = out.view(batch_size, q_seq_len, self.dim_v * self.num_heads)
        out = self.linear_out(out)
        return out, kv_cache


class GroupedQueryAttention(Attention):
    def __init__(self, config):
        super().__init__(config)
        assert self.num_group_heads is not None, "Number of group heads must be specified for GQA"
        assert self.num_heads % self.num_group_heads == 0, "Number of heads must be divisible by number of group heads."
        self.linear_q = nn.Linear(self.dim_model, self.dim_k * self.num_heads)
        self.linear_k = nn.Linear(self.dim_model, self.dim_k * self.num_group_heads)
        self.linear_v = nn.Linear(self.dim_model, self.dim_v * self.num_group_heads)
        self.linear_out = nn.Linear(self.dim_v * self.num_heads, self.dim_model)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Optional[Tensor] = None,
                kv_cache: Optional[Dict[str, Tensor]] = None):
        batch_size, q_seq_len, _ = queries.shape
        _, k_seq_len, _ = keys.shape

        q = self.linear_q(queries)
        q = q.view(batch_size, q_seq_len, self.num_heads, self.dim_k)
        q = q.permute(0, 2, 1, 3)

        k = self.linear_k(keys)
        k = k.view(batch_size, k_seq_len, self.num_group_heads, self.dim_k)
        k = k.permute(0, 2, 1, 3)

        v = self.linear_v(values)
        v = v.view(batch_size, k_seq_len, self.num_group_heads, self.dim_v)
        v = v.permute(0, 2, 1, 3)

        if kv_cache is not None:
            assert q_seq_len == k_seq_len == 1

            if "k" in kv_cache and "v" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v

        repeats = self.num_heads // self.num_group_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        scores = scores / (self.dim_k ** 0.5)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, float('-inf'))

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_seq_len, k.shape[2], dtype=torch.bool, device=scores.device),
                                     diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, v)
        out = out.permute(0, 2, 1, 3)
        out = out.contiguous()
        out = out.view(batch_size, q_seq_len, self.dim_v * self.num_heads)
        out = self.linear_out(out)
        return out, kv_cache


class MultiLatentAttention(Attention):
    def __init__(self, config):
        super().__init__(config)
        self.latent_dim = config.latent_dim
        self.linear_q = nn.Linear(self.dim_model, self.dim_k * self.num_heads)
        self.latent_k_proj = nn.Linear(self.dim_model, self.latent_dim * self.num_heads)
        self.latent_v_proj = nn.Linear(self.dim_model, self.latent_dim * self.num_heads)
        self.k_reconstruction = nn.Linear(self.latent_dim, self.dim_k, bias=False)
        self.v_reconstruction = nn.Linear(self.latent_dim, self.dim_v, bias=False)
        self.linear_out = nn.Linear(self.dim_v * self.num_heads, self.dim_model)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Optional[Tensor] = None,
                kv_cache: Optional[Dict[str, Tensor]] = None):
        batch_size, q_seq_len, _ = queries.shape
        _, k_seq_len, _ = keys.shape

        q = self.linear_q(queries)
        q = q.view(batch_size, q_seq_len, self.num_heads, self.dim_k)
        q = q.permute(0, 2, 1, 3)

        if kv_cache is not None:
            if "latent_k" in kv_cache and "latent_v" in kv_cache:
                latent_k_new = self.latent_k_proj(keys)
                latent_k_new = latent_k_new.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
                latent_k_new = latent_k_new.permute(0, 2, 1, 3)
                latent_k = torch.cat([kv_cache["latent_k"], latent_k_new], dim=2)

                latent_v_new = self.latent_v_proj(values)
                latent_v_new = latent_v_new.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
                latent_v_new = latent_v_new.permute(0, 2, 1, 3)
                latent_v = torch.cat([kv_cache["latent_v"], latent_v_new], dim=2)
            else:
                latent_k = self.latent_k_proj(keys)
                latent_k = latent_k.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
                latent_k = latent_k.permute(0, 2, 1, 3)

                latent_v = self.latent_v_proj(values)
                latent_v = latent_v.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
                latent_v = latent_v.permute(0, 2, 1, 3)
            kv_cache["latent_k"] = latent_k
            kv_cache["latent_v"] = latent_v
        else:
            latent_k = self.latent_k_proj(keys)
            latent_k = latent_k.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
            latent_k = latent_k.permute(0, 2, 1, 3)

            latent_v = self.latent_v_proj(values)
            latent_v = latent_v.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
            latent_v = latent_v.permute(0, 2, 1, 3)

        k = self.k_reconstruction(latent_k)
        v = self.v_reconstruction(latent_v)

        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        scores = scores / (self.dim_k ** 0.5)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, float('-inf'))

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_seq_len, k.shape[2], dtype=torch.bool, device=scores.device),
                                     diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, v)
        out = out.permute(0, 2, 1, 3)
        out = out.contiguous()
        out = out.view(batch_size, q_seq_len, self.dim_v * self.num_heads)
        out = self.linear_out(out)
        return out, kv_cache


class EfficientMultiLatentAttention(Attention):
    def __init__(self, config):
        super().__init__(config)
        self.low_dim = config.low_dim
        self.Wdq = nn.Parameter(torch.randn(self.dim_model, self.dim_model))
        self.Wuq = nn.Parameter(torch.randn(self.dim_model, self.dim_model))
        self.Wqr = nn.Parameter(torch.randn(self.dim_model, self.dim_k))
        self.Wdkv = nn.Parameter(torch.randn(self.dim_model, self.low_dim))
        self.Wkr = nn.Parameter(torch.randn(self.dim_model, self.dim_model))
        self.fused_upsample = nn.Parameter(torch.randn(self.low_dim, self.dim_model * 2))
        self.Wo = nn.Parameter(torch.randn(self.dim_model, self.dim_model))

    def forward(self, x: Tensor, kv_cache: Optional[Dict[str, Tensor]] = None):
        batch_size, T, _ = x.shape

        current_ckv = x @ self.Wdkv
        current_kr_proj = x @ self.Wkr
        current_upsampled = current_ckv @ self.fused_upsample

        if kv_cache is not None:
            if "ckv" in kv_cache:
                kv_cache["ckv"] = torch.cat([kv_cache["ckv"], current_ckv], dim=1)
                kv_cache["kr_proj"] = torch.cat([kv_cache["kr_proj"], current_kr_proj], dim=1)
                kv_cache["upsampled_kv"] = torch.cat([kv_cache["upsampled_kv"], current_upsampled], dim=1)
            else:
                kv_cache["ckv"] = current_ckv
                kv_cache["kr_proj"] = current_kr_proj
                kv_cache["upsampled_kv"] = current_upsampled
        else:
            kv_cache = {
                "ckv": current_ckv,
                "kr_proj": current_kr_proj,
                "upsampled_kv": current_upsampled
            }

        seq_len = kv_cache["ckv"].shape[1]
        Cq = x @ self.Wdq
        Qc = Cq @ self.Wuq
        Qr_proj = Cq @ self.Wqr

        Qc = Qc.view(batch_size, 1, self.num_heads, self.dim_k)
        Qc = Qc.permute(0, 2, 1, 3)

        Qr_proj = Qr_proj.view(batch_size, 1, self.num_heads, self.dim_k)
        Qr_proj = Qr_proj.permute(0, 2, 1, 3)

        q = torch.cat([Qc, Qr_proj], dim=-1)

        k_upsampled, v = torch.split(kv_cache["upsampled_kv"], self.dim_model, dim=-1)

        Kc = k_upsampled.view(batch_size, seq_len, self.num_heads, self.dim_k)
        Kc = Kc.permute(0, 2, 1, 3)

        Kr_proj_reshaped = kv_cache["kr_proj"].unsqueeze(2)
        Kr_proj_reshaped = Kr_proj_reshaped.repeat(1, 1, self.num_heads, 1)
        Kr_proj_reshaped = Kr_proj_reshaped.permute(0, 2, 1, 3)

        k = torch.cat([Kc, Kr_proj_reshaped], dim=-1)

        v = v.view(batch_size, seq_len, self.num_heads, self.dim_k)
        v = v.permute(0, 2, 1, 3)

        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = attn_scores / (self.dim_k * 2) ** 0.5

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(1, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v

        output = attn_output.permute(0, 2, 1, 3)
        output = output.reshape(batch_size, 1, self.dim_model)
        output = output @ self.Wo

        return output, kv_cache