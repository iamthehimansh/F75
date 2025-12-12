import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,embd_dim):
        super().__init__()
        self.W_k=nn.Linear(embd_dim,embd_dim)
        self.W_q=nn.Linear(embd_dim,embd_dim)
        self.W_v=nn.Linear(embd_dim,embd_dim)
    
    def forward(self,x):
        b,t,d=x.shape
        Q=self.W_q(x)
        K=self.W_k(x)
        V=self.W_v(x)

        score =Q@ K.transpose(-2,-1)
        score = score/(x.size(-1)**0.5)

        # masking future
        mask=torch.triu(torch.ones(t,t),diagonal=1).bool()
        score=score.masked_fill(mask,-1e10)

        weight = torch.softmax(score,dim=-1)
        out = weight @ V

        return out #, weight


class MultiHeadAttention_(nn.Module):
    def __init__(self,embd_dim,num_head):
        super().__init__()
        assert embd_dim %num_head ==0, "Number of head must devide Embedding dim"
        self.num_head= num_head
        self.head_dim = embd_dim // num_head

        self.W_q=nn.Linear(embd_dim,embd_dim)
        self.W_k=nn.Linear(embd_dim,embd_dim)
        self.W_v=nn.Linear(embd_dim,embd_dim)

        self.W_o=nn.Linear(embd_dim,embd_dim)

    def forward(self,x):
        b,t,d= x.shape

        Q=self.W_q(x)
        K=self.W_k(x)
        V=self.W_v(x)

        #             b,t,d =>b,t,n_h,h_d --> b,n_h,t,h_d
        Q=Q.view(b,t,self.num_head,self.head_dim).transpose(1,2)
        K=K.view(b,t,self.num_head,self.head_dim).transpose(1,2)
        V=V.view(b,t,self.num_head,self.head_dim).transpose(1,2)



        score= Q @ K.transpose(-2,-1)
        score = score/(d**0.5)

        mask= torch.triu(torch.ones(t,t),diagonal=1).bool()
        score=score.masked_fill(mask,-1e10)


        weight= torch.softmax(score,dim=1)

        out= weight @ V ## (b, n_h, t, head_dim)

        out=out.transpose(1,2).contiguous().view(b,t,d)

        out=self.W_o(out)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,embd_dim,num_head,attention_dropout=0,projection_dropout=0):
        super().__init__()

        assert embd_dim%num_head==0, "num head must devide embd dim"
        self.num_head=num_head
        self.head_dim=embd_dim//num_head

        self.W_q=nn.Linear(embd_dim,embd_dim)
        self.W_k=nn.Linear(embd_dim,embd_dim)
        self.W_v=nn.Linear(embd_dim,embd_dim)

        self.W_o=nn.Linear(embd_dim,embd_dim)

        self.attn_drop=nn.Dropout(attention_dropout)
        self.proj_drop=nn.Dropout(projection_dropout)

    def forward(self,x):
        b,t,d=x.shape

        Q=self.W_q(x)
        K=self.W_v(x)
        V=self.W_v(x)

        Q=Q.view(b,t,self.num_head,self.head_dim).transpose(1,2)
        K=K.view(b,t,self.num_head,self.head_dim).transpose(1,2)
        V=V.view(b,t,self.num_head,self.head_dim).transpose(1,2)


        score=Q@K.transpose(-2,-1)
        score=score/(d**0.50)

        mask=torch.triu(torch.ones(t,t),diagonal=1).bool().to(x.device)
        score=score.masked_fill(mask,-1e10)

        weight=torch.softmax(score,dim=-1)
        weight=self.attn_drop(weight)

        atten_droped=weight@V

        atten_droped=atten_droped.transpose(1,2).contiguous().view(b,t,d)

        out=self.W_o(atten_droped)
        projected=self.proj_drop(out)

        return projected


if __name__=="__main__":
    sa=MultiHeadAttention(64,4)
    rnd=torch.rand((1,2,64))
    print(rnd.shape)
    print(sa(rnd))