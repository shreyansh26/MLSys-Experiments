import torch
from triton.testing import do_bench
import matplotlib.pyplot as plt
import numpy as np

def lora_loop(
    y: torch.Tensor,        # [B, 1, out]
    x: torch.Tensor,        # [B, 1, in]
    A: torch.Tensor,        # [L, in, r]
    B: torch.Tensor,        # [L, r, out]
    I: torch.LongTensor,    # [B]
):
    for i, idx in enumerate(I.cpu().numpy()):
        y[i] += x[i] @ A[idx] @ B[idx]
    return y

def lora_cheat_bmm(
    y: torch.Tensor,        # [B, 1, out]
    x: torch.Tensor,        # [B, 1, in]
    cheat_A: torch.Tensor,  # [B, in, r]
    cheat_B: torch.Tensor,  # [B, r, out]
):
    """
    Assumes LoRA adapters are already concatenated. Sort of ideal case.
    """
    y += x @ cheat_A @ cheat_B
    return y

def bench_lora(type: str):
    num_loras = 50
    r = 16
    h1 = 4096
    h2 = 12288

    timings = []

    A = torch.randn(num_loras, h1, r, dtype=torch.float16, device="cuda")
    B = torch.randn(num_loras, r, h2, dtype=torch.float16, device="cuda")

    for bs in range(2, 64, 2):
        x = torch.randn(bs, 1, h1, dtype=torch.float16, device="cuda")
        y = torch.randn(bs, 1, h2, dtype=torch.float16, device="cuda")
        I = torch.randint(num_loras, (bs,), dtype=torch.long, device="cuda")

        # Group LoRA adapters in the right order.
        cheat_A = A[I, :, :]
        cheat_B = B[I, :, :]

        if type == "loop":
            timings.append(do_bench(lambda: lora_loop(y, x, A, B, I), quantiles=[0.5, 0.2, 0.8]))
        elif type == "cheat_bmm":
            timings.append(do_bench(lambda: lora_cheat_bmm(y, x, cheat_A, cheat_B), quantiles=[0.5, 0.2, 0.8]))
        else:
            raise ValueError(f"Invalid type: {type}")

    return timings

if __name__ == "__main__":
    lora_loop_timings = bench_lora("loop")
    lora_cheat_bmm_timings = bench_lora("cheat_bmm")

    plt.figure(figsize=(10, 5))

    la = np.asarray(lora_loop_timings)
    x_lora = np.array(range(2, 64, 2))
    yerr_lora = np.vstack([la[:, 0] - la[:, 1], la[:, 2] - la[:, 0]])
    plt.errorbar(x_lora, la[:, 0], yerr=yerr_lora, fmt="-o", color="#1f77b4", capsize=3, label="Lora loop median±range")

    fa = np.asarray(lora_cheat_bmm_timings)
    x_full = np.array(range(2, 64, 2))
    yerr_full = np.vstack([fa[:, 0] - fa[:, 1], fa[:, 2] - fa[:, 0]])
    plt.errorbar(x_full, fa[:, 0], yerr=yerr_full, fmt="-o", color="#ff7f0e", capsize=3, label="Cheat BMM median±range")

    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.xticks(x_lora)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/lora_loop_vs_cheat_bmm.png", dpi=200)