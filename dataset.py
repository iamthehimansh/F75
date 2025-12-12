import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=64):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + 1 + self.block_size]
        return x, y
