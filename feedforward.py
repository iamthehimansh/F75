import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self,embd_dim,mlp_ratio=4,dropout=0):
        super().__init__()
        hidden=int(embd_dim*mlp_ratio)
        self.net=nn.Sequential(
            nn.Linear(embd_dim,hidden),
            nn.GELU(),
            nn.Linear(hidden,embd_dim)
        )

        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.dropout(self.net(x))
