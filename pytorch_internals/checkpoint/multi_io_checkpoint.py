import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class MultiIOModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(30, 5)
        
    def checkpoint_fn(self, x1, x2, x3):
        # First branch
        h1 = self.linear1(x1)
        h1 = torch.relu(h1)
        
        # Second branch
        h2 = self.linear2(x2)
        h2 = torch.relu(h2)
        
        # Combine with third input
        h3 = torch.cat([h1, h2, x3], dim=1)  # 10 + 20 + 30 = 60
        out = self.linear3(h3[:, :30])  # Take first 30 dims
        
        return h1, h2, out  # Return multiple outputs
    
    def forward(self, x1, x2, x3, use_checkpoint=True):
        if use_checkpoint:
            return checkpoint(self.checkpoint_fn, x1, x2, x3, use_reentrant=False)
        return self.checkpoint_fn(x1, x2, x3)

def main():
    model = MultiIOModel()
    
    # Create inputs with different sizes
    x1 = torch.randn(32, 10, requires_grad=True)
    x2 = torch.randn(32, 20, requires_grad=True)
    x3 = torch.randn(32, 30, requires_grad=True)
    
    # Without checkpoint
    h1, h2, out = model(x1, x2, x3, use_checkpoint=False)
    loss = out.sum() + h1.sum() + h2.sum()
    loss.backward()
    grads_no_checkpoint = [x1.grad.clone(), x2.grad.clone(), x3.grad.clone()]
    
    # Reset grads
    x1.grad = x2.grad = x3.grad = None
    
    # With checkpoint
    h1, h2, out = model(x1, x2, x3, use_checkpoint=True)
    loss = out.sum() + h1.sum() + h2.sum()
    loss.backward()
    grads_checkpoint = [x1.grad, x2.grad, x3.grad]
    
    # Verify gradients match
    print("Gradients match:")
    for i, (g1, g2) in enumerate(zip(grads_no_checkpoint, grads_checkpoint)):
        print(f"Input {i+1}:", torch.allclose(g1, g2))

if __name__ == "__main__":
    main() 