import torch
import torch.nn as nn
from torch.optim import AdamW

def train(model, loader,tokenizer, epochs=5, lr=3e-4, device="cuda",):
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:     # x,y: (B, T)
            x = x.to(device)
            y = y.to(device)

            logits = model(x)              # (B, T, vocab)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss / len(loader):.4f}")

        # quick sample
        if (epoch + 1) % 1 == 0:
            start = torch.randint(0, model.lm_head.out_features, (1, 20)).to(device)
            generated = model.generate(start, max_new_tokens=60, temperature=0.8, top_k=40)
            decoded = tokenizer.decode(generated[0].cpu().tolist())
            print("Sample:", decoded[:200])
