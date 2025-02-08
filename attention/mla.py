import torch
from torch import nn
from torch import Tensor
from typing import Optional, Dict

class MultiLatentAttention(nn.Module):
    def __init__(self, config, causal = True):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim_model = config.dim_model
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v
        self.latent_dim = config.latent_dim

        self.linear_q = nn.Linear(config.dim_model, config.dim_k * config.num_heads)
        #can we compress the queries too?

        self.latent_k_proj = nn.Linear(config.dim_model, self.latent_dim * config.num_heads)
        self.latent_v_proj = nn.Linear(config.dim_model, self.latent_dim * config.num_heads)

        self.k_reconstruction = nn.Linear(self.latent_dim, config.dim_k)
        self.v_reconstruction = nn.Linear(self.latent_dim, config.dim_v)

        self.linear_out = nn.Linear(config.dim_v * config.num_heads, config.dim_model)

    def forward(self,
                queries:Tensor,
                keys:Tensor,
                values:Tensor,
                mask: Optional[Tensor] = None,
                kv_cache: Optional[Dict[str, Tensor]] = None):
        batch_size, q_seq_len, _ = queries.shape
        _, k_seq_len, _ = keys.shape
        _, v_seq_len, _ = values.shape

        q = self.linear_q(queries).view(batch_size, q_seq_len, self.num_heads, self.dim_k)
        q = q.permute(0,2,1,3)

        if kv_cache is not None:
            assert k_seq_len == 1, "if using KV Cache then k_seq_len must be 1"
            if "k" in kv_cache and "v" in kv_cache:
                cached_k, cached_v = kv_cache["k"], kv_cache["v"]
                cached_latent_k, cached_latent_v = kv_cache["latent_k"], kv_cache["latent_v"]
                new_latent_k = self.latent_k_proj(keys)
                new_latent_v = self.latent_v_proj(values)

                new_k = new_latent_k.view(batch_size, q_seq_len, self.num_heads, self.latent_dim)
                new_v = new_latent_v.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
                #TODO: A user could choose to lose track of reconstructed kv to safe memory
                new_k = self.k_reconstruction(new_k).permute(0,2,1,3)
                new_v = self.k_reconstruction(new_v).permute(0,2,1,3)

                kv_cache["k"] = torch.cat((cached_k, new_k), dim=2)
                kv_cache["v"] = torch.cat((cached_v, new_v), dim=2)

                kv_cache["latent_k"] = torch.cat((cached_latent_k, new_latent_k), dim=2)
                kv_cache["latent_v"] = torch.cat((cached_latent_v, new_latent_v), dim=2)

            else:
                latent_k = self.latent_k_proj(keys)
                latent_v = self.latent_v_proj(values)

                latent_k = latent_k.view(batch_size, q_seq_len, self.num_heads, self.latent_dim)
                latent_v = latent_v.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)

                latent_k = latent_k.permute(0,2,1,3)
                latent_v = latent_v.permute(0,2,1,3)

                kv_cache = {"k": keys, "v": values, "latent_k": latent_k, "latent_v": latent_v}

            k, v = kv_cache['k'], kv_cache['v']

        else:
            latent_k = self.latent_k_proj(keys)
            latent_v = self.latent_v_proj(values)

            latent_k = latent_k.view(batch_size, k_seq_len, self.num_heads, self.latent_dim)
            latent_v = latent_v.view(batch_size, v_seq_len, self.num_heads, self.latent_dim)

            k = self.k_reconstruction(latent_k).permute(0,2,1,3)
            v = self.v_reconstruction(latent_v).permute(0,2,1,3)

        q_k = torch.einsum("bnqd, bnkd -> bnqk", q,k) / (self.dim_k ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            q_k.masked_fill_(mask, -float('inf'))

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_seq_len, k_seq_len, dtype = torch.bool,
                                                device = q_k.device), diagonal = 1)
            q_k.masked_fill_(causal_mask, -float('inf'))

        attn_weights = torch.softmax(q_k, dim = -1)
        attn_values = torch.einsum("bnqk, bnvd -> bnqd", attn_weights, v)
        attn_values = (attn_values.permute(0,2,1,3).contiguous().view(batch_size, v_seq_len, self.dim_v * self.num_heads))
        y = self.linear_out(attn_values)
        return  y, kv_cache
