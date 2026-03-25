import torch
import tiktoken
from tiny_gpt import GPT, GPTConfig

# ============================
# Config — must match training
# ============================
enc = tiktoken.get_encoding("cl100k_base")
base_vocab_size = enc.n_vocab
eos_token_id = base_vocab_size
vocab_size = base_vocab_size + 1

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=64,
    n_layer=4,
    n_head=4,
    n_embd=256,
    eos_token_id=eos_token_id
)

# ============================
# Load weights
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(config).to(device)
model.load_state_dict(torch.load("runs/checkpoints/gpt_final.pt", map_location=device))
model.eval()
print("Model loaded. Type a prompt (Ctrl+C to quit).\n")

# ============================
# Interactive generation loop
# ============================
while True:
    try:
        prompt = input("Prompt: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        break

    if not prompt:
        continue

    context = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        out = model.generate(context, max_new_tokens=100, temperature=1.0, top_k=50)

    out_ids = out[0].tolist()
    clean_ids = [t for t in out_ids if t != eos_token_id]
    print("\n" + enc.decode(clean_ids) + "\n")
