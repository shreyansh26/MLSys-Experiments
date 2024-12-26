# Based on https://huggingface.co/blog/train_memory
import torch
from transformers import AutoModelForCausalLM

# Start recording memory snapshot history
torch.cuda.memory._record_memory_history(max_entries=100000)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for _ in range(3):
    inputs = torch.randint(0, 100, (16, 256), device="cuda")  # Dummy input
    loss = torch.mean(model(inputs).logits)  # Dummy loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Dump memory snapshot history to a file and stop recording
torch.cuda.memory._dump_snapshot("profile_llm.pkl")
torch.cuda.memory._record_memory_history(enabled=None)