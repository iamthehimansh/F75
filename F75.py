import torch
import torch.nn as nn
from transformer import Transformer
from embedding import PositionalEmbedding
from tokenization import CharacterTokenization

class F75(nn.Module):
    def __init__(self,embd_dim=32,num_head=4,attn_drop=0,proj_drop=0,dropout=0,mlp_ratio=4,block_size=64,n_layre=8,vocab_size=60):
        super().__init__()
        self.blocks=nn.ModuleList([
            Transformer(embd_dim,num_head,attn_drop,proj_drop,dropout,mlp_ratio)
            for _ in range(n_layre)
        ])

        self.final_lyrenorme=nn.LayerNorm(embd_dim)
        self.embdmodel=PositionalEmbedding(vocab_size,embd_dim,block_size)

        self.lm_head=nn.Linear(embd_dim,vocab_size,bias=False)
        self.block_size=block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def forward(self,input_ids):
        x=self.embdmodel(input_ids)
        for block in self.blocks:
            x=block(x)
        x=self.final_lyrenorme(x)
        logits=self.lm_head(x)
        return logits

    @torch.inference_mode()
    def generate(self,input_ids,max_new_tokens=50,temperature: int=1,top_k:int=1):
        
        device=input_ids.device

        for _ in range(max_new_tokens):
            input_ids= input_ids[:,-self.block_size:]
            logits=self.forward(input_ids)
            logits=logits[:,-1,:]/(temperature if temperature>0 else 1.0)

            if top_k is not None:
                top_k=min(max(1,top_k),logits.size(-1))
                v,_=torch.topk(logits,top_k)
                min_values=v[:,-1].unsqueeze(-1)
                logits=torch.where(logits<min_values,torch.full_like(logits,-1e10),logits)
            
            prob=torch.softmax(logits,dim=1)
            next_token=torch.multinomial(prob,num_samples=1)
            input_ids=torch.cat([input_ids,next_token],dim=1)
        return input_ids


if __name__ == "__main__":
    vocab_size=60
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = F75().to(device)
    tokenizer=CharacterTokenization()
    tokenizer.import_token()

    # dummy batch (B=2, T=10) random tokens
    batch = torch.randint(0, vocab_size, (2, 10), dtype=torch.long).to(device)
    logits = model(batch)  # (2, 10, vocab_size)
    print("Logits shape:", logits.shape)

    # quick generate (start tokens)
    start = torch.randint(0, vocab_size, (1, 8), dtype=torch.long).to(device)
    out = model.generate(start, max_new_tokens=16, temperature=0.8, top_k=50)
    print("Generated shape:", out.shape)
    print(out)
    out_decoded=tokenizer.decode(out[0].to("cpu").tolist())
    print(out_decoded)

