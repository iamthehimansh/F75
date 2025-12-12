import torch.nn as nn
import torch

class PositionalEmbedding(nn.Module):
    def __init__(self,vocab_size=60,embd_dim=32,block_size=64):
        super().__init__()

        self.token_emb=nn.Embedding(vocab_size,embd_dim)
        self.pos_embd=nn.Embedding(block_size,embd_dim)

    def forward(self,x):
        b,t=x.shape
        pos=torch.arange(t,device=x.device)

        pos_embd=self.pos_embd(pos)
        token_embd=self.token_emb(x)
        return pos_embd+token_embd
    # (batch, seq_len, embed_dim)
    
if __name__=="__main__":
    emb = PositionalEmbedding()
    y = emb(torch.tensor([[1,2]]))
    print(y.shape) 