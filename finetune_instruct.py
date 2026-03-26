import os
import torch
import torch.nn.functional as F
import tiktoken
from torch.utils.tensorboard import SummaryWriter
from tiny_gpt import GPT, GPTConfig
from instruction_data import INSTRUCTION_DATA

# ============================
# Tokenizer & special tokens
# ============================
enc = tiktoken.get_encoding("cl100k_base")
base_vocab_size = enc.n_vocab
eos_token_id = base_vocab_size
vocab_size = base_vocab_size + 1

block_size = 64   # matches pretraining

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=4,
    n_head=4,
    n_embd=256,
    eos_token_id=eos_token_id
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Load pretrained weights for language prior.
# Without them the model can't form coherent words (100K vocab, 10 samples).
# We use a tiny LR so the language prior is nudged, not overwritten.
# ============================
model = GPT(config).to(device)
ckpt_path = "runs/checkpoints/gpt_final.pt"
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
print(f"Loaded pretrained weights from {ckpt_path}")

# Very low LR — we only want to steer, not retrain
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

os.makedirs("runs/checkpoints", exist_ok=True)
writer = SummaryWriter(log_dir="runs/instruct_finetune")

# ============================
# Encode dataset
# Format: "Question: ...\nAnswer: ...<EOS>"
# Loss only on the Answer portion (question labels = -100).
# Build causal LM pairs explicitly: x = tokens[:-1], y = labels[1:].
# ============================
def encode_sample(q, a):
    q_ids = enc.encode(f"Question: {q}\nAnswer: ")
    a_ids = enc.encode(a) + [eos_token_id]
    input_ids = q_ids + a_ids
    labels    = [-100] * len(q_ids) + a_ids
    return input_ids, labels


def pad_or_truncate(ids, length, pad_id=0):
    return (ids + [pad_id] * length)[:length]


sequence_length = block_size + 1
all_inputs, all_labels = [], []
for item in INSTRUCTION_DATA:
    inp, lbl = encode_sample(item["question"], item["answer"])
    inp = pad_or_truncate(inp, sequence_length)
    lbl = pad_or_truncate(lbl, sequence_length, pad_id=-100)
    all_inputs.append(inp[:-1])
    all_labels.append(lbl[1:])

inputs_tensor = torch.tensor(all_inputs, dtype=torch.long)
labels_tensor = torch.tensor(all_labels, dtype=torch.long)
N = len(all_inputs)
valid_targets = int((labels_tensor != -100).sum().item())
print(f"Dataset: {N} samples, block_size={block_size}, supervised_tokens={valid_targets}")


def get_batch(batch_size=4):
    idx = torch.randint(0, N, (batch_size,))
    return inputs_tensor[idx].to(device), labels_tensor[idx].to(device)


# ============================
# Fine-tuning loop
# With only 10 examples, stop once the answers are learned rather than
# pushing for a very low loss that encourages memorization artifacts.
# ============================
num_steps   = 3000
batch_size  = 4
early_stop  = 0.30
print_every = 100

for step in range(num_steps):
    model.train()
    x, y = get_batch(batch_size)

    logits, _ = model(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=-100
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar("Loss/instruct", loss.item(), step)

    if step % print_every == 0:
        print(f"step={step:4d}  loss={loss.item():.4f}")

        model.eval()
        prompt = "Question: What is youth?\nAnswer: "
        ctx = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model.generate(ctx, max_new_tokens=50, temperature=0.2, top_k=5, repetition_penalty=1.0)
        out_ids = [t for t in out[0].tolist() if t != eos_token_id]
        answer = enc.decode(out_ids).split("Answer:")[-1].strip()
        print(f"  >> {answer}\n")

    if loss.item() < early_stop:
        print(f"Early stop at step={step}, loss={loss.item():.4f}")
        torch.save(model.state_dict(), "runs/checkpoints/instruct_final.pt")
        break

torch.save(model.state_dict(), "runs/checkpoints/instruct_final.pt")
writer.close()
print("Done. Saved to runs/checkpoints/instruct_final.pt")
