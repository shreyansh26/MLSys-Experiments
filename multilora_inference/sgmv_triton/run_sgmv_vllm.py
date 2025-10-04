
# SPDX-License-Identifier: Apache-2.0
# Host-side test harness for sgmv_triton.sgmv_triton (no extra deps).
#
import torch
import random
import math
import sgmv_triton as sgmv

torch.manual_seed(0)
random.seed(0)

def _make_problem(batch=64, F_IN=1024, F_OUT=2048, R=16, num_layers=2, num_lora=8, seq_len=1, dtype=torch.float16):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = batch if seq_len == 1 else batch * seq_len

    # Stacks: T = num_lora * num_layers
    T = num_lora * num_layers
    A = torch.randn(T, F_IN, R, device=device, dtype=dtype)
    B = torch.randn(T, F_OUT, R, device=device, dtype=dtype)

    # Inputs
    if seq_len == 1:
        X = torch.randn(batch, F_IN, device=device, dtype=dtype)
        Y = torch.zeros(batch, F_OUT, device=device, dtype=dtype)
        # Choose indices with some sentinels
        indices = torch.randint(0, num_lora + 1, (batch,), device=device, dtype=torch.int32)
        indices_rows = indices  # [B]
    else:
        X = torch.randn(batch, seq_len, F_IN, device=device, dtype=dtype)
        Y = torch.zeros(batch, seq_len, F_OUT, device=device, dtype=dtype)
        indices = torch.randint(0, num_lora + 1, (batch,), device=device, dtype=torch.int32)
        # Broadcast per-seq to rows
        indices_rows = indices.repeat_interleave(seq_len)  # [B*seq_len]
        X = X.reshape(batch * seq_len, F_IN)
        Y = Y.reshape(batch * seq_len, F_OUT)

    return X, Y, A, B, indices_rows, indices, device

def _reference(Y, X, A, B, indices_rows, num_layers, layer_idx, num_lora, scale, accumulate):
    # Compute reference on device in float32 for correctness
    N, F_IN = X.shape
    F_OUT = Y.shape[1]
    T, Fin, R = A.shape
    Y_ref = torch.zeros((N, F_OUT), dtype=torch.float32, device=X.device)
    if accumulate:
        Y_ref += Y.float()
    for n in range(N):
        aid = int(indices_rows[n].item())
        if aid == num_lora:
            continue  # sentinel => no contribution
        key = aid * num_layers + layer_idx
        z = torch.matmul(X[n].float(), A[key].float())  # [R]
        y = torch.matmul(z, B[key].float().t())         # [F_OUT]
        Y_ref[n] += scale * y
    return Y_ref.to(dtype=Y.dtype)

def run_demo():
    # 2D tests
    print("SGMV 2D (accumulate=False)")
    X, Y, A, B, indices_rows, _, device = _make_problem(batch=64, F_IN=768, F_OUT=1536, R=32,
                                                        num_layers=2, num_lora=8, seq_len=1, dtype=torch.float16)
    Y0 = Y.clone()
    sgmv.sgmv_triton(Y, X, A, B, indices_rows,
                     num_layers=2, layer_idx=1, num_lora_adapters=8,
                     scale=1.0, accumulate=False)
    Y_ref = _reference(Y0, X, A, B, indices_rows, 2, 1, 8, 1.0, False)
    err = (Y.float() - Y_ref.float()).abs().max().item()
    print(f"max_err={err:.3e}")

    print("SGMV 2D (accumulate=True)")
    Y = Y0.clone()
    # Add some base to test accumulate path
    base = torch.randn_like(Y)
    Y += base
    Y_ref = base + _reference(Y0, X, A, B, indices_rows, 2, 1, 8, 1.0, False)
    sgmv.sgmv_triton(Y, X, A, B, indices_rows,
                     num_layers=2, layer_idx=1, num_lora_adapters=8,
                     scale=1.0, accumulate=True)
    err = (Y.float() - Y_ref.float()).abs().max().item()
    print(f"max_err (accumulate)={err:.3e}\n")

    # 3D test (seq_len > 1)
    print("SGMV 3D (accumulate=False)")
    X, Y, A, B, indices_rows, indices, device = _make_problem(batch=8, F_IN=512, F_OUT=1024, R=16,
                                                              num_layers=3, num_lora=6, seq_len=5, dtype=torch.float16)
    Y0 = Y.clone()
    sgmv.sgmv_triton(Y, X, A, B, indices_rows,
                     num_layers=3, layer_idx=0, num_lora_adapters=6,
                     scale=0.5, accumulate=False)
    Y_ref = _reference(Y0, X, A, B, indices_rows, 3, 0, 6, 0.5, False)
    err = (Y.float() - Y_ref.float()).abs().max().item()
    print(f"max_err 3D={err:.3e}")

if __name__ == "__main__":
    run_demo()
