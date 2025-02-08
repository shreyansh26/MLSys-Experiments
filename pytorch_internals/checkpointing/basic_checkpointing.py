import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def __init__(self, size=1024, depth=16):
        super(LargeModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(size, size * 4),  # Expand
                nn.ReLU(),
                nn.Linear(size * 4, size),  # Contract
                nn.LayerNorm(size)
            ) for _ in range(depth)
        ])

    def _block(self, x, layers):
        h = x
        for layer in layers:
            h = layer(h)
            h = h * torch.sigmoid(h)
            h = h + torch.tanh(h)
        return h

    def forward_without_checkpoint(self, x):
        return self._block(x, self.layers)

    def forward_with_checkpoint(self, x, chunk_size=2):
        # Divide the layers into chunks and wrap each chunk with checkpoint.
        for i in range(0, len(self.layers), chunk_size):
            chunk = self.layers[i:i+chunk_size]
            x = checkpoint(lambda inp: self._block(inp, chunk), x, use_reentrant=False)
        return x

def run_iteration(model, x, use_ckpt):
    if use_ckpt:
        out = model.forward_with_checkpoint(x)
    else:
        out = model.forward_without_checkpoint(x)
    loss = out.sum()
    loss.backward()
    return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Increase batch size or model depth if you’d like to see larger differences.
    batch_size = 512
    model = LargeModel(size=1024, depth=16).to(device)
    model.train()  # Must be in training mode to keep activations for grad.
    x_temp = torch.randn(batch_size, 1024, requires_grad=True, device=device)
    # Run without checkpointing repeatedly
    iterations = 5
    
    print("\nRunning without checkpointing:")
    for i in range(iterations):
        x_temp.grad = None
        run_iteration(model, x_temp, use_ckpt=False)
        torch.cuda.synchronize()
    mem_no_ckpt = torch.cuda.max_memory_allocated()
    print(f"Peak memory (no checkpoint): {mem_no_ckpt / 1e9:.2f} GB")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Run with checkpointing repeatedly
    print("\nRunning with checkpointing:")
    for i in range(iterations):
        x_temp.grad = None
        run_iteration(model, x_temp, use_ckpt=True)
        torch.cuda.synchronize()
    mem_ckpt = torch.cuda.max_memory_allocated()
    print(f"Peak memory (with checkpoint): {mem_ckpt / 1e9:.2f} GB")
    
    # Print memory summary for further insights
    print("\nMemory Summary:")
    print(torch.cuda.memory_summary())

if __name__ == "__main__":
    main()