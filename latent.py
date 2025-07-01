import torch
import math
from torch import nn, Tensor
from typing import Optional, Dict

class MultiLatentAttention(nn.Module):
    def __init__(self,
                 d_model:int,
                 n_heads:int,
                 latent_dim:int,
                 dim_q:int =None,
                 dim_kv:int = None,
                 ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert self.dim_kv % 2 == 0, "dim_kv must   be a multiple of 2"

        self.q_proj_down = nn.Linear(self.d_model, latent_dim)
        self.q_proj_up = nn.Linear(latent_dim, self.d_model)
        self.q_layer_norm = nn.LayerNorm(latent_dim)

        self.kv_proj_down = nn.Linear(self.d_model, latent_dim)
        self.kv_proj_up = nn.Linear(latent_dim, 2 * self.d_model)
        self.kv_layer_norm = nn.LayerNorm(latent_dim)

        self.linear_out = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, kv_cache: Optional[Dict[str, Tensor]] = None, mask = None):
        batch_size, seq_len, _ = x.shape

        queries = self.q_proj_down(x) #compress queries
        queries = self.q_layer_norm(queries)

        queries = self.q_proj_up(x) #decompress queries, we dont need rope here
        if self.training is False:
            if kv_cache is not None:
                compressed_kv = self.kv_proj_down(x)
                compressed_kv = self.kv_layer_norm(compressed_kv)
                cached_compressed_kv = kv_cache['kv']
                full_compressed_kv = torch.concat([cached_compressed_kv, compressed_kv],
                                                  dim = 1)
                kv_cache['kv'] = full_compressed_kv
            else:
                kv_cache = dict()
                compressed_kv = self.kv_proj_down(x)
                compressed_kv = self.kv_layer_norm(compressed_kv)
                kv_cache['kv'] = compressed_kv
            K_V = kv_cache['kv']

        else:
            compressed_kv = self.kv_proj_down(x) #when we train we update the weights of the proj dim and up
            compressed_kv = self.kv_layer_norm(compressed_kv)
            K_V = compressed_kv

        K_V = self.kv_proj_up(K_V) #decompress everything
        keys, values = torch.split(K_V, self.d_model, dim = 1)

        queries = queries.view(batch_size, seq_len, self.n_heads, self.d_model //self.n_heads)
        queries = queries.permute(0,2,1,3)

        keys = keys.view(batch_size, seq_len, self.n_heads, self.d_model //self.n_heads)
        keys = keys.permute(0,2,1,3)

        values = values.view(batch_size, seq_len, self.n_heads, self.d_model //self.n_heads)
        values = values.permute(0,2,1,3)

        attn_output = torch.einsum("bnqd, bnkd -> bnqk", queries, keys)
        attn_output = attn_output / math.sqrt(self.dim_kv // 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_output.masked_fill_(mask, -float('inf'))

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype = torch.bool,
                                            device = attn_output.device), diagonal = 1)
        attn_output.masked_fill_(causal_mask, -float('inf'))
        attn_output = torch.softmax(attn_output, dim=-1)

        attn_output = torch.einsum("bnqk, bnkd -> bnqd", attn_output, values)
        attn_output = attn_output.permute(0,2,1,3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.d_model)
        y = self.linear_out(attn_output)
        return y, kv_cache







