from torch import nn
from feedforward import FeedForwardConfig

class FeedForward(nn.Module):
    def __init__(self, config:FeedForwardConfig):
        super().__init__()
        self.dim_model = config.dim_model
        self.dim_ff = config.dim_ff
        self.dropout = config.dropout

class StandardFeedForward(FeedForward):
    def __init__(self, config):
        super().__init__(config)
        self.linear1 = nn.Linear(self.dim_model, self.dim_ff)
        self.linear2 = nn.Linear(self.dim_ff, self.dim_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout_layer(self.activation(self.linear1(x))))
