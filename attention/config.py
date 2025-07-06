from dataclasses import dataclass
from typing import Optional
@dataclass
class AttentionConfig:
    dim_ff: int
    dim_k: int
    dim_v: int
    num_heads: int
    dim_model:int
    is_causal: bool
    dropout: float = 0.1
    latent_dim:Optional[int] = None
    num_group_heads:Optional[int] = None
    low_dim:Optional[int] = None


