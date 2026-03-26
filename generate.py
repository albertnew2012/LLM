import torch
import tiktoken
from tiny_gpt import GPT, GPTConfig
import os

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
# Load weights — prefer instruction-tuned if available
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
instruct_ckpt = "runs/checkpoints/instruct_final.pt"
pretrain_ckpt = "runs/checkpoints/gpt_final.pt"
ckpt = instruct_ckpt if os.path.exists(instruct_ckpt) else pretrain_ckpt

model = GPT(config).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
model.eval()
print(f"Model loaded from {ckpt}")
print("Type your question (Ctrl+C to quit).\n")

# ============================
# Interactive generation loop
# Wraps user input in instruction format: "Question: ...\nAnswer: "
# ============================
while True:
    try:
        question = input("Question: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        break

    if not question:
        continue

    prompt = f"Question: {question}\nAnswer: "
    context = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        out = model.generate(
            context,
            max_new_tokens=80,
            temperature=0.2,
            top_k=5,
            repetition_penalty=1.0,
        )

    out_ids = [t for t in out[0].tolist() if t != eos_token_id]
    # Print only the answer portion
    full_text = enc.decode(out_ids)
    answer = full_text.split("Answer:")[-1].strip()
    print(f"\nAnswer: {answer}\n")
