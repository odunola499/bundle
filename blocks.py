from torch import nn
from attention.attention import MHA, MQA, MLA, EfficientMLA
from feedforward.feedforward import StandardFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = self._get_attention_layer(config.attention)
        self.ff = self._get_ff_layer(config.ff)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def _get_attention_layer(self, attn_config):
        if attn_config.name == "MHA":
            return MHA(attn_config)
        elif attn_config.name == "MQA":
            return MQA(attn_config)
        elif attn_config.name == "MLA":
            return MLA(attn_config)
        elif attn_config.name == "EfficientMLA":
            return EfficientMLA(attn_config)
        else:
            raise ValueError(f"Unknown attention type: {attn_config.name}")

    def _get_ff_layer(self, ff_config):
        if ff_config.name == "StandardFeedForward":
            return StandardFeedForward(ff_config)
        else:
            raise ValueError(f"Unknown feed-forward type: {ff_config.name}")

    def forward(self, x, mask=None, kv_cache=None):
        attn_output, kv_cache = self.attention(x, x, x, mask, kv_cache)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x, kv_cache

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = self._get_attention_layer(config.attention)
        self.cross_attention = self._get_attention_layer(config.attention)
        self.ff = self._get_ff_layer(config.ff)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def _get_attention_layer(self, attn_config):
        if attn_config.name == "MHA":
            return MHA(attn_config)
        elif attn_config.name == "MQA":
            return MQA(attn_config)
        elif attn_config.name == "MLA":
            return MLA(attn_config)
        elif attn_config.name == "EfficientMLA":
            return EfficientMLA(attn_config)
        else:
            raise ValueError(f"Unknown attention type: {attn_config.name}")

    def _get_ff_layer(self, ff_config):
        if ff_config.name == "StandardFeedForward":
            return StandardFeedForward(ff_config)
        else:
            raise ValueError(f"Unknown feed-forward type: {ff_config.name}")

    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None, kv_cache=None):
        self_attn_output, kv_cache = self.self_attention(x, x, x, self_attn_mask, kv_cache)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        cross_attn_output, _ = self.cross_attention(x, enc_output, enc_output, cross_attn_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x, kv_cache
