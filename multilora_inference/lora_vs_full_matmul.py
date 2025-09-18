import torch
from triton.testing import do_bench
import matplotlib.pyplot as plt
import numpy as np

# Lora vs Full matmul
def lora_matmul():
    """
    Assumes ideal case of LoRA adapters grouped in the right order.
    """
    r = 16
    h1 = 4096
    h2 = 12288
   
    timings = []

    for bs in range(2, 64, 2):
        x = torch.randn(bs, 1, h1, dtype=torch.float16, device="cuda")
        A = torch.randn(bs, h1, r, dtype=torch.float16, device="cuda")
        B = torch.randn(bs, r, h2, dtype=torch.float16, device="cuda")

        timings.append(do_bench(lambda: x @ A @ B, quantiles=[0.5, 0.2, 0.8]))

    return timings

def full_matmul():
    h1 = 4096
    h2 = 12288
    
    timings = []

    for bs in range(2, 64, 2):
        x = torch.randn(bs, 1, h1, dtype=torch.float16, device="cuda")
        W = torch.randn(h1, h2, dtype=torch.float16, device="cuda")
    
        timings.append(do_bench(lambda: x @ W, quantiles=[0.5, 0.2, 0.8]))

    return timings

if __name__ == "__main__":
    lora_timings = lora_matmul()
    full_timings = full_matmul()

    plt.figure(figsize=(10, 5))

    la = np.asarray(lora_timings)
    x_lora = np.array(range(2, 64, 2))
    yerr_lora = np.vstack([la[:, 0] - la[:, 1], la[:, 2] - la[:, 0]])
    plt.errorbar(x_lora, la[:, 0], yerr=yerr_lora, fmt="-o", color="blue", capsize=3, label="Lora median±range")

    fa = np.asarray(full_timings)
    x_full = np.array(range(2, 64, 2))
    yerr_full = np.vstack([fa[:, 0] - fa[:, 1], fa[:, 2] - fa[:, 0]])
    plt.errorbar(x_full, fa[:, 0], yerr=yerr_full, fmt="-o", color="orange", capsize=3, label="Full median±range")

    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.xticks(x_lora)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/lora_vs_full_matmul.png", dpi=200)