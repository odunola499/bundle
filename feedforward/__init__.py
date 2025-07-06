from .feedforward import FeedForward
from dataclasses import dataclass

@dataclass
class FeedForwardConfig:
    dim_ff:int
    dim_model:int
    dropout:int