import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import time

class RNNWithCheckpoint(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create RNN cells manually for fine-grained control
        self.cells = nn.ModuleList([
            nn.RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
    
    def forward_step(self, x, hidden_states):
        new_states = []
        current_input = x
        
        for i, cell in enumerate(self.cells):
            new_hidden = cell(current_input, hidden_states[i])
            new_states.append(new_hidden)
            current_input = new_hidden
            
        return current_input, new_states
    
    def forward(self, x, use_checkpoint=True):
        # x shape: [seq_len, batch_size, input_size]
        seq_len, batch_size, input_size = x.shape
        device = x.device
        
        # Initialize hidden states
        hidden_states = [torch.zeros(batch_size, self.hidden_size, device=device) 
                        for _ in range(self.num_layers)]
        outputs = []
        
        for t in range(seq_len):
            if use_checkpoint:
                # Checkpoint each timestep
                def step_fn(input_t, *prev_states):
                    out, new_states = self.forward_step(input_t, list(prev_states))
                    return (out,) + tuple(new_states)
                
                step_outputs = checkpoint(step_fn, x[t], *hidden_states, use_reentrant=False)
                output = step_outputs[0]
                hidden_states = list(step_outputs[1:])
                hidden_states = [h.detach() for h in hidden_states]
            else:
                output, hidden_states = self.forward_step(x[t], hidden_states)
            outputs.append(output)
            
        return torch.stack(outputs)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Model dimensions
    seq_len, batch_size, input_size = 1024, 32, 1024
    hidden_size, num_layers = 4096, 10
    
    # Create model and input
    model = RNNWithCheckpoint(input_size, hidden_size, num_layers).to(device)
    x = torch.randn(seq_len, batch_size, input_size, requires_grad=True, device=device)
    
    def run_and_time(use_checkpoint):
        start = time.time()
        
        out = model(x, use_checkpoint=use_checkpoint)
        loss = out.sum()
        loss.backward()
        
        elapsed = time.time() - start
        return elapsed
    
    iterations = 5
    # Run without checkpoint
    for i in range(iterations):
        time_normal = run_and_time(False)
    peak_mem_normal = torch.cuda.max_memory_allocated() / 1e6
    print(f"\nWithout checkpoint:")
    print(f"Time: {time_normal:.3f}s")
    print(f"Peak Memory: {peak_mem_normal:.2f}MB")
        
    # Reset gradients
    x.grad = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run with checkpoint
    for i in range(iterations):
        time_checkpointing = run_and_time(True)
    peak_mem_checkpointing = torch.cuda.max_memory_allocated() / 1e6
    print(f"\nWith checkpoint:")
    print(f"Time: {time_checkpointing:.3f}s")
    print(f"Peak Memory: {peak_mem_checkpointing:.2f}MB")
        
    print(f"\nMemory saved: {(peak_mem_normal - peak_mem_checkpointing):.2f}MB")
    print(f"Time overhead: {((time_checkpointing/time_normal) - 1)*100:.1f}%")

if __name__ == "__main__":
    main() 