import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

class LargeModel(nn.Module):
    def __init__(self, size=1024, depth=16, use_checkpoint=True):
        super().__init__()
        layers = []
        for _ in range(depth):
            # Define a block that expands, transforms, and contracts the features.
            block = nn.Sequential(
                nn.Linear(size, size * 4),    # Expand
                nn.ReLU(),
                nn.Linear(size * 4, size),    # Contract
                nn.LayerNorm(size),
            )
            # Wrap the block if we want checkpointing.
            if use_checkpoint:
                # Here we choose the non-reentrant variant (recommended) but you can adjust via CheckpointImpl.
                block = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

    # A single forward method remains unchanged regardless of checkpointing.
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
            # Additional computation outside the wrapped modules remains as is.
            h = h * torch.sigmoid(h)
            h = h + torch.tanh(h)
        return h + x

# Helper function to run one training iteration.
def run_iteration(model, x):
    out = model(x)
    loss = out.sum()
    loss.backward()
    return loss

def main():
    # Select device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    batch_size = 512
    iterations = 5

    # Run without checkpointing repeatedly
    print("\nRunning without checkpointing:")
    model_no_ckpt = LargeModel(size=1024, depth=16, use_checkpoint=False).to(device)
    model_no_ckpt.train()
    for i in range(iterations):
        x_temp = torch.randn(batch_size, 1024, requires_grad=True, device=device)
        run_iteration(model_no_ckpt, x_temp)
        if device.type == "cuda":
            torch.cuda.synchronize()
    mem_no_ckpt = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    print(f"Peak memory (no checkpoint): {mem_no_ckpt / 1e9:.2f} GB")
    
    # Reset peak memory stats
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    del model_no_ckpt

    # Run with checkpointing repeatedly
    print("\nRunning with checkpointing:")
    model_ckpt = LargeModel(size=1024, depth=16, use_checkpoint=True).to(device)
    model_ckpt.train()
    for i in range(iterations):
        x_temp = torch.randn(batch_size, 1024, requires_grad=True, device=device)
        run_iteration(model_ckpt, x_temp)
        if device.type == "cuda":
            torch.cuda.synchronize()
    mem_ckpt = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    print(f"Peak memory (with checkpoint): {mem_ckpt / 1e9:.2f} GB")

    if device.type == "cuda":
        print("\nMemory Summary:")
        print(torch.cuda.memory_summary())

if __name__ == "__main__":
    main()