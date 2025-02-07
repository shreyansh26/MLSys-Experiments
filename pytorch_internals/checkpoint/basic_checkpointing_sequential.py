import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import functools

class LargeModelWithCheckpoint(nn.Module):
    def __init__(self, size=1024, depth=16, segments=4):
        """
        Initializes a model made of a sequence of blocks.
        
        Args:
            size (int): The input/output dimension.
            depth (int): The number of sequential blocks.
            segments (int): The number of segments to split the model into for checkpointing.
                            (More segments means smaller chunks, which saves more memory but recomputes more.)
        """
        super(LargeModelWithCheckpoint, self).__init__()
        
        # Build a list of modules (blocks). For each block we do a Linear -> ReLU -> Linear -> LayerNorm operation.
        blocks = []
        for _ in range(depth):
            block = nn.Sequential(
                nn.Linear(size, size * 4),
                nn.ReLU(),  # non-linearity
                nn.Linear(size * 4, size),
                nn.LayerNorm(size)
            )
            # Optionally, if you want extra element-wise operations (e.g. mixing activations)
            # you can wrap them inside a custom Module and add to the block.
            blocks.append(block)
        
        # We convert the list of blocks into an nn.Sequential.
        self.blocks = nn.Sequential(*blocks)
        self.segments = segments

    def forward(self, x):
        """
        In training mode, we use checkpoint_sequential to save
        memory by not storing intermediate activations.
        In evaluation mode, we simply do a normal forward pass.
        """
        if self.training:
            # checkpoint_sequential splits self.blocks into self.segments pieces and
            # applies checkpointing over them. This call is equivalent to:
            # cp.checkpoint(checkpointed_function_segment, x) for each segment.
            return checkpoint_sequential(self.blocks, self.segments, x, use_reentrant=False)
        else:
            return self.blocks(x)

class LargeModelWithoutCheckpoint(nn.Module):
    def __init__(self, size=1024, depth=16, segments=4):
        """
        Initializes a model made of a sequence of blocks.
        
        Args:
            size (int): The input/output dimension.
            depth (int): The number of sequential blocks.
            segments (int): The number of segments to split the model into for checkpointing.
                            (More segments means smaller chunks, which saves more memory but recomputes more.)
        """
        super(LargeModelWithoutCheckpoint, self).__init__()
        
        # Build a list of modules (blocks). For each block we do a Linear -> ReLU -> Linear -> LayerNorm operation.
        blocks = []
        for _ in range(depth):
            block = nn.Sequential(
                nn.Linear(size, size * 4),
                nn.ReLU(),  # non-linearity
                nn.Linear(size * 4, size),
                nn.LayerNorm(size)
            )
            # Optionally, if you want extra element-wise operations (e.g. mixing activations)
            # you can wrap them inside a custom Module and add to the block.
            blocks.append(block)
        
        # We convert the list of blocks into an nn.Sequential.
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        In training mode, we use checkpoint_sequential to save
        memory by not storing intermediate activations.
        In evaluation mode, we simply do a normal forward pass.
        """
        if self.training:
            return self.blocks(x)
        else:
            return self.blocks(x)
        
def run_iteration(model, x):
    out = model(x)
    loss = out.sum()
    loss.backward()
    return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Increase batch size or model depth if you’d like to see larger differences.
    batch_size = 512
    model_without_checkpoint = LargeModelWithoutCheckpoint(size=1024, depth=16).to(device)
    model_without_checkpoint.train()  # Must be in training mode to keep activations for grad.

    x_temp = torch.ones(batch_size, 1024, requires_grad=True, device=device)
    iterations = 5
    # Run without checkpointing repeatedly
    print("\nRunning without checkpointing:")
    for i in range(iterations):
        x_temp.grad = None
        run_iteration(model_without_checkpoint, x_temp)
        torch.cuda.synchronize()
    mem_no_ckpt = torch.cuda.max_memory_allocated()
    print(f"Peak memory (no checkpoint): {mem_no_ckpt / 1e9:.2f} GB")
    x_grad_no_ckpt = x_temp.grad.clone()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    del model_without_checkpoint

    model_with_checkpoint = LargeModelWithCheckpoint(size=1024, depth=16).to(device)
    model_with_checkpoint.train()  # Must be in training mode to keep activations for grad.
    # Run with checkpointing repeatedly
    print("\nRunning with checkpointing:")
    for i in range(iterations):
        x_temp.grad = None
        run_iteration(model_with_checkpoint, x_temp)
        torch.cuda.synchronize()
    mem_ckpt = torch.cuda.max_memory_allocated()
    print(f"Peak memory (with checkpoint): {mem_ckpt / 1e9:.2f} GB")
    x_grad_ckpt = x_temp.grad.clone()
    
    # Print memory summary for further insights
    print("\nMemory Summary:")
    print(torch.cuda.memory_summary())
    print(f"Gradients match: {torch.allclose(x_grad_no_ckpt, x_grad_ckpt)}")

if __name__ == "__main__":
    main()