import torch
import torch.nn as nn
from selfatemtion import MultiHeadAttention
from feedforward import FeedForward

class Transformer(nn.Module):
    def __init__(self,embd_dim,num_head,attn_drop=0,proj_drop=0,dropout=0,mlp_ratio=4):
        super().__init__()
        self.larenorm1=nn.LayerNorm(embd_dim)
        self.multiheadatten=MultiHeadAttention(embd_dim,num_head,attn_drop,proj_drop)
        self.larenorm2=nn.LayerNorm(embd_dim)
        self.feedforward=FeedForward(embd_dim,mlp_ratio,dropout)
    def forward(self,x):
        x=x+self.multiheadatten(self.larenorm1(x))
        x=x+self.feedforward(self.larenorm2(x))

        return x