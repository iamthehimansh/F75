import torch
from F75 import F75
from tokenization import CharacterTokenization
from dataset import TextDataset
from train import train
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"



# load tokenizer (already has vocab of 60)
tokenizer = CharacterTokenization()
tokenizer.import_token()

# dataset + dataloader
block_size = 64
batch_size = 32


# model
model = F75(
    embd_dim=32,
    num_head=4,
    attn_drop=0,
    proj_drop=0,
    dropout=0,
    mlp_ratio=4,
    block_size=64,
    n_layre=8,
    vocab_size=61
).to(device)


state = torch.load("f75_shakespeare.pt", map_location=device)
model.load_state_dict(state)

model.eval()

def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f"Total parameters: {count_params(model):,}")
print("Model loaded successfully!")


def genrate(text_org):
    text=tokenizer.encode(text_org)
    text=torch.tensor(text).unsqueeze(0).to(device)
    out=model.generate(text,max_new_tokens=500,temperature=2)
    out=out[0].to("cpu").tolist()
    decoded=tokenizer.decode(out)
    return text_org+"".join(decoded)

while True:
    inp=input("U: ")
    out=genrate(inp)
    print(f"A: {out}")