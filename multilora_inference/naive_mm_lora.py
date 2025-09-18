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

def lora_gbmm(
    y: torch.Tensor,        # [B, 1, out]
    x: torch.Tensor,        # [B, 1, in]
    A: torch.Tensor,        # [L, in, r]
    B: torch.Tensor,        # [L, r, out]
    I: torch.LongTensor,    # [B]
):
    A_grouped = torch.index_select(A, 0, I)
    B_grouped = torch.index_select(B, 0, I)
    y += x @ A_grouped @ B_grouped
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
        elif type == "gbmm":
            timings.append(do_bench(lambda: lora_gbmm(y, x, A, B, I), quantiles=[0.5, 0.2, 0.8]))
        else:
            raise ValueError(f"Invalid type: {type}")

    return timings

if __name__ == "__main__":
    lora_loop_timings = bench_lora("loop")
    lora_cheat_bmm_timings = bench_lora("cheat_bmm")
    lora_gbmm_timings = bench_lora("gbmm")

    plt.figure(figsize=(10, 5))

    la = np.asarray(lora_loop_timings)
    x_lora_loop = np.array(range(2, 64, 2))
    yerr_lora = np.vstack([la[:, 0] - la[:, 1], la[:, 2] - la[:, 0]])
    plt.errorbar(x_lora_loop, la[:, 0], yerr=yerr_lora, fmt="-o", color="blue", capsize=3, label="Lora loop median±range")

    lc = np.asarray(lora_cheat_bmm_timings)
    x_lora_cheat_bmm = np.array(range(2, 64, 2))
    yerr_full = np.vstack([lc[:, 0] - lc[:, 1], lc[:, 2] - lc[:, 0]])
    plt.errorbar(x_lora_cheat_bmm, lc[:, 0], yerr=yerr_full, fmt="-o", color="orange", capsize=3, label="Cheat BMM median±range")

    lg = np.asarray(lora_gbmm_timings)
    x_lora_gbmm = np.array(range(2, 64, 2))
    yerr_full = np.vstack([lg[:, 0] - lg[:, 1], lg[:, 2] - lg[:, 0]])
    plt.errorbar(x_lora_gbmm, lg[:, 0], yerr=yerr_full, fmt="-o", color="red", capsize=3, label="GBMM median±range")

    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.xticks(x_lora_loop)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/lora_loop_vs_cheat_bmm_vs_gbmm.png", dpi=200)