import torch
from torch import nn
from torch import Tensor
from typing import Optional, Dict



class DecoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        name = config.name
        if name not in ['MHA', 'MQA', 'MLA']:
            raise ValueError(f"Invalid attention type: {name}")
        attention = None
        if name == 'MHA':
            attention = MultiHeadAttention(config, causal = True)
        elif name == 'MQA':
            attention = MultiQueryAttention(config, causal = True)
        self.attention = attention

    def forward(self, x:Tensor, mask:Tensor = None, kv_cache:Dict[str, Tensor] = None):
        return self.attention.forward()

class EncoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        name = config.name
        if name not in ['MHA', 'MQA', 'MLA']:
            raise ValueError(f"Invalid attention type: {name}")
        attention = None
        if name == 'MHA':
            attention = MultiHeadAttention(config, causal = False)
        elif name == 'MQA':
            attention = MultiQueryAttention(config, causal = False)
        self.attention = attention

    def forward(self, x:Tensor, mask:Tensor = None, kv_cache:Dict[str, Tensor] = None):
        raise NotImplementedError

class MultiHeadAttention(nn.Module):
    def __init__(self, config, causal = True):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim_model = config.dim_model
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v
        self.is_causal = causal #used to decide if a model is causal

        self.linear_q = nn.Linear(config.dim_model, config.dim_k * config.num_heads)
        self.linear_k = nn.Linear(config.dim_model, config.dim_k * config.num_heads)
        self.linear_v = nn.Linear(config.dim_model, config.dim_v * config.num_heads)
        self.linear_out = nn.Linear(config.dim_v * config.num_heads, config.dim_model)

    def forward(self, queries:Tensor,
                keys:Tensor,
                values:Tensor,
                mask: Optional[Tensor] = None,  # [batch_size, q_seq_len]
                kv_cache:Optional[Dict[str, Tensor]] = None):
        batch_size, q_seq_len, q_hidden_dim = queries.shape
        _, k_seq_len, _ = keys.shape
        _, v_seq_len, _ = values.shape

        q = self.linear_q(queries)
        q = q.view(batch_size, q_seq_len, self.num_heads, self.dim_k).permute(0, 2, 1, 3)
        if kv_cache is not None:
            assert k_seq_len == 1, "Since KV Cache is enabled, k_seq_len must be 1"
            if "k" in kv_cache and "v" in kv_cache:
                cached_k, cached_v = kv_cache["k"], kv_cache["v"]
                new_k = self.linear_k(keys)
                new_v = self.linear_v(values)

                new_k = new_k.view(batch_size, q_seq_len, self.num_heads, self.dim_k).permute(0,2,1,3)
                new_v = new_v.view(batch_size, k_seq_len, self.num_heads, self.dim_v).permute(0,2,1,3)

                kv_cache["k"] = torch.cat((cached_k, new_k), dim=2)
                kv_cache["v"] = torch.cat((cached_v, new_v), dim=2)

            else:
                k = self.linear_k(keys)
                v = self.linear_v(values)

                k = k.view(batch_size, k_seq_len, self.num_heads, self.dim_k).permute(0,2,1,3)
                v = v.view(batch_size, v_seq_len, self.num_heads, self.dim_v).permute(0,2,1,3)

                kv_cache = {"k": k, "v": v}
            k,v = kv_cache['k'], kv_cache['v']


        else:
            k = self.linear_k(keys)
            v = self.linear_v(values)

            k = k.view(batch_size, k_seq_len, self.num_heads, self.dim_k).permute(0, 2, 1, 3)
            v = v.view(batch_size, v_seq_len, self.num_heads, self.dim_v).permute(0, 2, 1, 3)

        q_k = torch.einsum("bnqd, bnkd -> bnqk", q, k) / (self.dim_k ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            q_k.masked_fill_(mask, -float('inf'))

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_seq_len, k_seq_len, dtype = torch.bool,
                                                device = q_k.device), diagonal = 1)
            q_k.masked_fill_(causal_mask, -float('inf'))

        attn_weights = torch.softmax(q_k, dim=-1)
        attn_values = torch.einsum("bnqk, bnvd -> bnqd", attn_weights, v)
        attn_values = (attn_values.permute(0,2,1,3)
                       .contiguous().view(batch_size, v_seq_len, self.dim_v * self.num_heads))
        y = self.linear_out(attn_values)
        return y, kv_cache


class MultiQueryAttention(nn.Module):
    def __init__(self, config, causal = True):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim_model = config.dim_model
        self.dim_k = config.dim_k
        self.dim_v = config.dim_v
        self.is_causal = causal

        self.linear_q = nn.Linear(config.dim_model, config.dim_k * config.num_heads)
        self.linear_k = nn.Linear(config.dim_model, config.dim_k) #same head for everything
        self.linear_v = nn.Linear(config.dim_model, config.dim_v) #same head for everything
        self.linear_out = nn.Linear(config.dim_v * config.num_heads, config.dim_model)

    def forward(self, queries:Tensor,
                keys:Tensor,
                values:Tensor,
                mask: Optional[Tensor] = None,  # [batch_size, q_seq_len]
                kv_cache: Optional[Dict[str, Tensor]] = None):
        batch_size, q_seq_len, _ = queries.shape
        _, k_seq_len, _ = keys.shape

        q = self.linear_q(queries).view(batch_size, q_seq_len, self.num_heads, self.dim_k).permute(0,2,1,3)

        if kv_cache is not None:
            assert k_seq_len == 1, "if using KV Cache then k_seq_len must be 1"
            if "k" in kv_cache and "v" in kv_cache:
                cached_k, cached_v = kv_cache["k"], kv_cache["v"]
                new_k = self.linear_k(keys)
                new_v = self.linear_v(values)

                new_k = new_k.view(batch_size, q_seq_len, self.num_heads, self.dim_k).permute(0,2,1,3)
                new_v = new_v.view(batch_size, k_seq_len, self.num_heads, self.dim_v).permute(0,2,1,3)

                kv_cache["k"] = torch.cat((cached_k, new_k), dim=2)
                kv_cache["v"] = torch.cat((cached_v, new_v), dim=2)

            else:
                k = self.linear_k(keys)
                v = self.linear_v(values)

                k = k.view(batch_size, k_seq_len, self.num_heads, self.dim_k).permute(0,2,1,3)
                v = v.view(batch_size, k_seq_len, self.num_heads, self.dim_v).permute(0,2,1,3)

                kv_cache = {"k": k, "v": v}

            k,v = kv_cache['k'], kv_cache['v']

        else:
            k = self.linear_k(keys)
            v = self.linear_v(values)

            k = k.view(batch_size, k_seq_len, self.num_heads, self.dim_k).permute(0,2,1,3)
            v = v.view(batch_size, k_seq_len, self.num_heads, self.dim_v).permute(0,2,1,3)

        q_k = torch.einsum('bnqd,bnkd -> bnqk', q, k) / (self.dim_k ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            q_k.masked_fill_(mask, -float('inf'))

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_seq_len, k_seq_len, dtype = torch.bool,
                                                device = q_k.device), diagonal = 1)
            q_k.masked_fill_(causal_mask, -float('inf'))

        weights = torch.softwax(q_k, dim = -1)
        attn_values = torch.einsum('bnqk, bnvd -> bnqd', weights, v)
        return self.linear_out(attn_values), kv_cache


class GroupedQueryAttention(nn.Module):
    pass





