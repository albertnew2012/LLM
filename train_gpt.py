import os
import torch
import tiktoken
from tiny_gpt import GPT, GPTConfig

# ============================
# 1. Load data
# ============================
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 tokenizer

# Encode entire dataset
data = enc.encode(text)   # ← RETURNS list of token IDs
data = torch.tensor(data, dtype=torch.long)

print("Total tokens:", len(data))
print("Example:", data[:20])

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ============================
# 2. Data loader
# ============================
batch_size = 16
block_size = 64   # context length (in tokens, NOT characters)

def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+1+block_size] for i in ix])
    return x.cuda(), y.cuda()

# ============================
# 3. Model config (LARGE VOCAB)
# ============================
vocab_size = enc.n_vocab   # ~100,000 tokens
print("GPT-4 tokenizer vocab size:", vocab_size)

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=4,
    n_head=4,
    n_embd=256,
)

model = GPT(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# ============================
# 4. Training loop
# ============================
for step in range(2000):

    x, y = get_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step={step} loss={loss.item():.4f}")

        # Sample some text
        # context = torch.tensor([[enc.encode("time")[0]]], dtype=torch.long).cuda()
        # out = model.generate(context, max_new_tokens=50, top_k=40)
        # decoded = enc.decode(out[0].tolist())
        
        
        prompt = "When you feel old, remember that youth"
        context = torch.tensor([enc.encode(prompt)], dtype=torch.long).cuda()
        # Higher temperature → more inventive text
        # No top_k → less deterministic
        out = model.generate(context, max_new_tokens=80, temperature=1.2, top_k=50)
        decoded = enc.decode(out[0].tolist())
        print("=== SAMPLE ===")
        print(decoded)
        print("==============")