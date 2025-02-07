import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial

# A simple wrapper that mimics the internal checkpoint_wrapper.
class CheckpointWrapper(nn.Module):
    def __init__(self, module, use_reentrant=False, **ckpt_kwargs):
        super().__init__()
        # Save the module to be executed under checkpointing.
        self.module = module
        # Here we “bake in” the checkpoint function.
        # You might choose to pass additional kwargs to checkpoint.
        self.ckpt_fn = partial(checkpoint, use_reentrant=use_reentrant, **ckpt_kwargs)

    def forward(self, *args, **kwargs):
        # In many cases, you have to worry about kwargs with the reentrant version.
        # In this simplified example we assume pure positional arguments.
        # If you need to support kwargs (as in the real _checkpoint_wrapper),
        # you’d want to pack/unpack them.
        return self.ckpt_fn(self.module, *args, **kwargs)

# Your original large model, now with an option to wrap its layers.
class LargeModel(nn.Module):
    def __init__(self, size=1024, depth=16, checkpoint_layers=False):
        super().__init__()
        layers = []
        for _ in range(depth):
            # Define a per-layer module.
            layer = nn.Sequential(
                nn.Linear(size, size * 4),  # Expand
                nn.ReLU(),
                nn.Linear(size * 4, size),  # Contract
                nn.LayerNorm(size)
            )
            # Optionally wrap with checkpointing.
            if checkpoint_layers:
                # Wrap the layer so that its forward call is executed under checkpoint.
                layer = CheckpointWrapper(layer, use_reentrant=False)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    # Single forward function
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
            # Additional operations can be performed outside the wrapped module.
            h = h * torch.sigmoid(h)
            h = h + torch.tanh(h)
        return h + x

# Helper that runs one iteration; no need to create two forward styles.
def run_iteration(model, x):
    out = model(x)
    loss = out.sum()
    loss.backward()
    return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    batch_size = 512
    iterations = 5

    # Run without checkpointing repeatedly
    print("\nRunning without checkpointing:")
    model_no_ckpt = LargeModel(size=1024, depth=16, checkpoint_layers=False).to(device)
    model_no_ckpt.train()
    for i in range(iterations):
        x_temp = torch.randn(batch_size, 1024, requires_grad=True, device=device)
        run_iteration(model_no_ckpt, x_temp)
        torch.cuda.synchronize()
    mem_no_ckpt = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    print(f"Peak memory (no checkpoint): {mem_no_ckpt / 1e9:.2f} GB")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    del model_no_ckpt

    # Run with checkpointing repeatedly
    print("\nRunning with checkpointing:")
    model_ckpt = LargeModel(size=1024, depth=16, checkpoint_layers=True).to(device)
    model_ckpt.train()
    for i in range(iterations):
        x_temp = torch.randn(batch_size, 1024, requires_grad=True, device=device)
        run_iteration(model_ckpt, x_temp)
        torch.cuda.synchronize()
    mem_ckpt = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    print(f"Peak memory (with checkpoint): {mem_ckpt / 1e9:.2f} GB")

    if device.type == "cuda":
        print("\nMemory Summary:")
        print(torch.cuda.memory_summary())

if __name__ == "__main__":
    main()