import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from nexus_llm import NexusLLM

# 1. Dataset Generation
DATA_PATH = "dataset.txt"

samples = [
    "User: Print Hi\nBot: ```python\nprint(\"Hi\")\n```",
    "User: generate a python code to print Hi\nBot: ```python\nprint(\"Hi\")\n```",
    "User: write python code to calculate factorial\nBot: ```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n\nprint(factorial(5))\n```",
    "User: Search about Vande Bharat\nBot: Vande Bharat Express is a high-speed inter-city train operated by Indian Railways.",
    "User: Who are you?\nBot: I am OLIT Nexus, a custom neural language model.",
    "User: Hello\nBot: Hello! How can I assist you today?",
]

with open(DATA_PATH, "w", encoding="utf-8") as f:
    for _ in range(400):
        for sample in samples:
            f.write(sample + "\n<|endoftext|>\n")

# 2. Train Tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.BPEDecoder()

trainer = trainers.BpeTrainer(
    special_tokens=["<|pad|>", "<|unk|>", "<|startoftext|>", "<|endoftext|>"],
    vocab_size=3000
)
tokenizer.train(files=[DATA_PATH], trainer=trainer)
tokenizer.save("tokenizer.json")

# 3. Model Training
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text_data = f.read()

tokens = tokenizer.encode(text_data).ids
data_tensor = torch.tensor(tokens, dtype=torch.long)

vocab_size = tokenizer.get_vocab_size()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = NexusLLM(vocab_size=vocab_size, dim=256, n_head=8, n_layer=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

batch_size = 16
seq_len = 128
epochs = 400

print(f"Training Custom LLM on device: {device}...")
model.train()

for epoch in range(epochs):
    ix = torch.randint(len(data_tensor) - seq_len, (batch_size,))
    x = torch.stack([data_tensor[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data_tensor[i+1:i+seq_len+1] for i in ix]).to(device)

    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pt")
print("Training complete! Model saved to 'model.pt' and tokenizer to 'tokenizer.json'.")