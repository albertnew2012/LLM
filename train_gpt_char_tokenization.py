# train_tiny_gpt.py
import os
import torch
import torch.nn as nn
from tiny_gpt import GPT, GPTConfig

# =====================
# 1. Config / Hyperparams
# =====================
batch_size = 64
block_size = 128      # context length
max_iters = 20000
eval_interval = 200
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.1

# =====================
# 2. Data loading (char-level)
# =====================
here = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(here, "input.txt")

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded dataset with {len(text)} characters")

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join(itos[i] for i in tokens)


data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """
    Returns:
      x: (B, T)
      y: (B, T)
    """
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + 1 + block_size] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# =====================
# 3. Model init
# =====================
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
)

model = GPT(config).to(device)
print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")


state_dict = torch.load("tiny_gpt_char_20000.pt")
model.load_state_dict(state_dict["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# =====================
# 4. Training loop
# =====================
for iter in range(max_iters):

    # Periodic evaluation
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # quick sample
        context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with token 0
        generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=20)
        print("=== sample ===")
        print(decode(generated[0].tolist()))
        print("==============")

    # Get batch
    x, y = get_batch("train")

    # Forward + backward + update
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print("Training done.")

# =====================
# 5. Save model
# =====================
ckpt_path = os.path.join(here, "tiny_gpt_char.pt")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": vars(config),
        "stoi": stoi,
        "itos": itos,
    },
    ckpt_path,
)
print(f"Saved checkpoint to {ckpt_path}")
