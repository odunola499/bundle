from torch import nn
import torch
from attention import MultiHeadAttention, AttentionConfig
from feedforward import FeedForward, FeedForwardConfig

from dataclasses import dataclass

@dataclass
class ModelConfig(AttentionConfig, FeedForwardConfig):
    dim_ff:int = 2048
    dim_k:int = 64
    dim_v:int = 64
    num_heads:int = 8
    dim_model:int = 512
    is_causal:bool = True


    dropout:float = 0.1
    max_seq_len:int = 512

    max_spatial_seq_len:int = 1024
    depth_seq_len:int= 8
    pad_id:int = 0

    spatial_layers:int = 8
    depth_layers:int = 8
    vocab_size:int = 1024

class TransformerBlock(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feedforward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x =  x + self.feedforward(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config:ModelConfig, depth = True):
        super().__init__()

        self.layer_norm = nn.LayerNorm(config.dim_model)

        if depth:
            num_layers = config.depth_layers
        else:
            num_layers = config.spatial_layers

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(config))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x




class RQTransformer(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()

        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim_model)

        self.spatial_start_token = nn.Parameter(torch.randn(config.dim_model))

        self.spatial_pos_emb = nn.Embedding(config.max_spatial_seq_len + 1 , config.dim_model)
        self.depth_pos_emb = nn.Embedding(config.depth_seq_len, config.dim_model)

        self.spatial_transformer = Transformer(config)
        self.depth_transformer = Transformer(config, depth = True)

        self.output = nn.Linear(config.dim_model, config.vocab_size)

    def forward(self, input_ids, ):
        #input_ids: [batch_size, seq_len, depth_seq_len]
        pass





if __name__ == "__main__":
    config = ModelConfig()
    model = RQTransformer(config)


