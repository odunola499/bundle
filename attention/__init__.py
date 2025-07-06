
from .attention import Attention, MultiHeadAttention, MultiQueryAttention, MultiLatentAttention, GroupedQueryAttention
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
            attention = MultiHeadAttention(config)
        elif name == 'MQA':
            attention = MultiQueryAttention(config)
        elif name == 'GQA':
            attention = GroupedQueryAttention(config)
        self.attention = attention

    def forward(self, x:Tensor,
                mask:Tensor = None,
                kv_cache:Optional[Dict[str, Tensor]] = None):
        return self.attention.forward(x,x,x ,mask, kv_cache)

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
        elif name == 'GQA':
            attention = GroupedQueryAttention(config, causal = False)
        self.attention = attention

    def forward(self, x:Tensor, mask:Tensor = None):
        return self.attention.forward(x,x,x ,mask)