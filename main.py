import torch
from F75 import F75
from tokenization import CharacterTokenization
from dataset import TextDataset
from train import train
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# load text
with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# load tokenizer (already has vocab of 60)
tokenizer = CharacterTokenization()
tokenizer.import_token()

# dataset + dataloader
block_size = 64
batch_size = 32

dataset = TextDataset(text, tokenizer, block_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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

# train
train(model, loader, epochs=10, lr=3e-4, device=device,tokenizer=tokenizer)


# save
torch.save(model.state_dict(), "f75_shakespeare.pt")

# test
text_org="The littel "
text=text_org
text=tokenizer.encode(text)
text=torch.tensor(text).unsqueeze(0).to(device)
out=model.generate(text,max_new_tokens=500,temperature=2)
out=out[0].to("cpu").tolist()
decoded=tokenizer.decode(out)
print(text_org+"".join(decoded),sep="")