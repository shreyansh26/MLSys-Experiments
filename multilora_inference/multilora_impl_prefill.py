import torch
from triton.testing import do_bench
import matplotlib.pyplot as plt
import numpy as np
from bgmv_cuda import lora_bgmv_cuda as lora_bgmv_cuda_impl
from bgmv_triton import lora_bgmv_triton as lora_bgmv_triton_impl
from sgmv_triton import lora_sgmv_triton as lora_sgmv_triton_impl

def lora_loop(
    y: torch.Tensor,        # [B, n, out]
    x: torch.Tensor,        # [B, n, in]
    A: torch.Tensor,        # [L, in, r]
    B: torch.Tensor,        # [L, r, out]
    I: torch.LongTensor,    # [B]
):
    for i, idx in enumerate(I.cpu().numpy()):
        y[i] += x[i] @ A[idx] @ B[idx]
    return y

def lora_cheat_bmm(
    y: torch.Tensor,        # [B, n, out]
    x: torch.Tensor,        # [B, n, in]
    cheat_A: torch.Tensor,  # [B, in, r]
    cheat_B: torch.Tensor,  # [B, r, out]
):
    """
    Assumes LoRA adapters are already concatenated. Sort of ideal case.
    """
    y += x @ cheat_A @ cheat_B
    return y

def lora_gbmm(
    y: torch.Tensor,        # [B, n, out]
    x: torch.Tensor,        # [B, n, in]
    A: torch.Tensor,        # [L, in, r]
    B: torch.Tensor,        # [L, r, out]
    I: torch.LongTensor,    # [B]
):
    A_grouped = torch.index_select(A, 0, I)
    B_grouped = torch.index_select(B, 0, I)
    y += x @ A_grouped @ B_grouped
    return y

def lora_bgmv_cuda(
    y: torch.Tensor,        # [B, n, out]
    x: torch.Tensor,        # [B, n, in]
    A_T: torch.Tensor,      # [L, r, in]
    B_T: torch.Tensor,      # [L, out, r]
    I: torch.LongTensor,    # [B]
):
    lora_bgmv_cuda_impl(y, x, A_T, B_T, I)
    return y

def lora_bgmv_triton(
    y: torch.Tensor,        # [B, n, out]
    x: torch.Tensor,        # [B, n, in]
    A_T: torch.Tensor,      # [L, r, in]
    B_T: torch.Tensor,      # [L, out, r]
    I: torch.LongTensor,    # [B]
):
    lora_bgmv_triton_impl(y, x, A_T, B_T, I)
    return y

def lora_sgmv_triton(
    y: torch.Tensor,        # [B, n, out]
    x: torch.Tensor,        # [B, n, in]
    A_T: torch.Tensor,      # [L, r, in]
    B_T: torch.Tensor,      # [L, out, r]
    I: torch.LongTensor,    # [B]
    num_lora_adapters: int,
):
    lora_sgmv_triton_impl(y, x, A_T, B_T, I, num_lora_adapters)
    return y


def run_multilora_test(type: str, dtype: torch.dtype = torch.float16, bench: bool = True):
    num_loras = 20
    r = 16
    h1 = 4096
    h2 = 16384
    n = 32

    timings = []

    if bench is False:
        dtype = torch.float32

    A = torch.randn(num_loras, h1, r, dtype=dtype, device="cuda")
    B = torch.randn(num_loras, r, h2, dtype=dtype, device="cuda")
    A_T = A.transpose(1, 2).contiguous()
    B_T = B.transpose(1, 2).contiguous()

    for bs in range(2, 64, 2):
        x = torch.randn(bs, n, h1, dtype=dtype, device="cuda")
        y = torch.randn(bs, n, h2, dtype=dtype, device="cuda")
        I = torch.randint(num_loras, (bs,), dtype=torch.long, device="cuda")

        # Group LoRA adapters in the right order.
        cheat_A = A[I, :, :]
        cheat_B = B[I, :, :]

        # Correctness check
        if not bench:
            y_loop = y.clone()
            y_gbmm = y.clone()
            y_bgmv_cuda = y.clone()
            y_bgmv_triton = y.clone()
            y_sgmv_triton = y.clone()
            y_loop_out = lora_loop(y_loop, x, A, B, I)
            y_gbmm_out = lora_gbmm(y_gbmm, x, A, B, I)
            y_bgmv_cuda_out = lora_bgmv_cuda(y_bgmv_cuda, x, A_T, B_T, I)
            y_bgmv_triton_out = lora_bgmv_triton(y_bgmv_triton, x, A_T, B_T, I)
            y_sgmv_triton_out = lora_sgmv_triton(y_sgmv_triton, x, A_T, B_T, I, num_loras)
            print(y_loop_out)
            print(y_gbmm_out)
            print(y_bgmv_cuda_out)
            print(y_bgmv_triton_out)
            print(y_sgmv_triton_out)
            torch.testing.assert_close(y_loop_out, y_gbmm_out, atol=1e-1, rtol=1e-1)
            torch.testing.assert_close(y_loop_out, y_bgmv_cuda_out, atol=1e-1, rtol=1e-1)
            torch.testing.assert_close(y_loop_out, y_bgmv_triton_out, atol=1e-1, rtol=1e-1)
            torch.testing.assert_close(y_loop_out, y_sgmv_triton_out, atol=1, rtol=1)
        if bench:
            if type == "loop":
                timings.append(do_bench(lambda: lora_loop(y, x, A, B, I), quantiles=[0.5, 0.2, 0.8]))
            elif type == "cheat_bmm":
                timings.append(do_bench(lambda: lora_cheat_bmm(y, x, cheat_A, cheat_B), quantiles=[0.5, 0.2, 0.8]))
            elif type == "gbmm":
                timings.append(do_bench(lambda: lora_gbmm(y, x, A, B, I), quantiles=[0.5, 0.2, 0.8]))
            elif type == "bgmv_cuda":
                timings.append(do_bench(lambda: lora_bgmv_cuda(y, x, A_T, B_T, I), quantiles=[0.5, 0.2, 0.8]))
            elif type == "bgmv_triton":
                timings.append(do_bench(lambda: lora_bgmv_triton(y, x, A_T, B_T, I), quantiles=[0.5, 0.2, 0.8]))
            elif type == "sgmv_triton":
                timings.append(do_bench(lambda: lora_sgmv_triton(y, x, A_T, B_T, I, num_loras), quantiles=[0.5, 0.2, 0.8]))
            else:
                raise ValueError(f"Invalid type: {type}")

    return timings

if __name__ == "__main__":
    print("Correctness check...")
    run_multilora_test("", dtype=torch.float32, bench=False)

    print("Starting benchmarks...")
    lora_loop_timings = run_multilora_test("loop")
    lora_cheat_bmm_timings = run_multilora_test("cheat_bmm")
    lora_gbmm_timings = run_multilora_test("gbmm")
    lora_bgmv_cuda_timings = run_multilora_test("bgmv_cuda")
    lora_bgmv_triton_timings = run_multilora_test("bgmv_triton")
    lora_sgmv_triton_timings = run_multilora_test("sgmv_triton")
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

    lbc = np.asarray(lora_bgmv_cuda_timings)
    x_lora_bgmv_cuda = np.array(range(2, 64, 2))
    yerr_full = np.vstack([lbc[:, 0] - lbc[:, 1], lbc[:, 2] - lbc[:, 0]])
    plt.errorbar(x_lora_bgmv_cuda, lbc[:, 0], yerr=yerr_full, fmt="-o", color="green", capsize=3, label="BGMV CUDA median±range")

    lbt = np.asarray(lora_bgmv_triton_timings)
    x_lora_bgmv_triton = np.array(range(2, 64, 2))
    yerr_full = np.vstack([lbt[:, 0] - lbt[:, 1], lbt[:, 2] - lbt[:, 0]])
    plt.errorbar(x_lora_bgmv_triton, lbt[:, 0], yerr=yerr_full, fmt="-o", color="purple", capsize=3, label="BGMV Triton median±range")

    lst = np.asarray(lora_sgmv_triton_timings)
    x_lora_sgmv_triton = np.array(range(2, 64, 2))
    yerr_full = np.vstack([lst[:, 0] - lst[:, 1], lst[:, 2] - lst[:, 0]])
    plt.errorbar(x_lora_sgmv_triton, lst[:, 0], yerr=yerr_full, fmt="-o", color="yellow", capsize=3, label="SGMV Triton median±range")

    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.xticks(x_lora_cheat_bmm)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/prefill_lora_loop_vs_cheat_bmm_vs_gbmm_vs_bgmv_cuda_triton_vs_sgmv_triton.png", dpi=200)