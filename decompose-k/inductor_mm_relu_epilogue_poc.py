"""Check whether Inductor fuses `torch.mm + relu` into a Triton epilogue.

Run this file normally. It re-launches itself with `TORCH_LOGS=output_code`,
compiles a simple `relu(mm(a, b))`, saves the generated-code log, and reports
whether the log shows a Triton kernel containing both `aten.mm` and `aten.relu`.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_dtype(name: str):
    import torch

    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def relu_mm(a, b):
    import torch

    return torch.relu(torch.mm(a, b))


def run_child(args: argparse.Namespace) -> None:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = parse_dtype(args.dtype)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)

    a = torch.randn((args.m, args.k), device="cuda", dtype=dtype)
    b = torch.randn((args.k, args.n), device="cuda", dtype=dtype)

    compiled = torch.compile(relu_mm, mode=args.mode, dynamic=args.dynamic)
    expected = relu_mm(a, b)
    actual = compiled(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)

    eager_ms = triton.testing.do_bench(lambda: relu_mm(a, b), warmup=10, rep=50)
    compiled_ms = triton.testing.do_bench(lambda: compiled(a, b), warmup=10, rep=50)

    print(f"torch={torch.__version__} triton={triton.__version__}", flush=True)
    print(f"device={torch.cuda.get_device_name()}", flush=True)
    print(
        f"shape=({args.m}, {args.k}) x ({args.k}, {args.n}) "
        f"dtype={args.dtype} mode={args.mode} dynamic={args.dynamic}",
        flush=True,
    )
    print(f"eager_ms={eager_ms:.6f}", flush=True)
    print(f"compiled_ms={compiled_ms:.6f}", flush=True)
    print(f"compiled_over_eager={compiled_ms / eager_ms:.3f}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    if args.force_exit:
        os._exit(0)


def analyze_output_code(log_text: str) -> tuple[str, list[str]]:
    fused_markers = [
        "Original ATen: [aten.mm, aten.relu]",
        "triton_helpers.maximum",
    ]
    fused = all(marker in log_text for marker in fused_markers) and (
        "triton_red_fused_mm_relu" in log_text
        or "triton_poi_fused_mm_relu" in log_text
        or "fused_mm_relu" in log_text
    )
    extern_mm = "extern_kernels.mm" in log_text or "extern_kernels.addmm" in log_text
    decompose_k = "decompose_k_mm_" in log_text and "extern_kernels.bmm_dtype" in log_text
    separate_relu = "Original ATen: [aten.relu]" in log_text and (
        "triton_poi_fused_relu" in log_text or "triton_per_fused_relu" in log_text
    )

    notes = []
    if fused:
        verdict = "fused_triton_mm_relu"
        notes.append("found a Triton kernel with source nodes [aten.mm, aten.relu]")
        notes.append("found triton_helpers.maximum before the store")
    elif decompose_k and separate_relu:
        verdict = "decompose_k_mm_plus_separate_relu"
        notes.append("found Inductor Decompose-K matmul")
        notes.append("found a separate relu pointwise kernel after Decompose-K")
    elif extern_mm and separate_relu:
        verdict = "extern_mm_plus_separate_relu"
        notes.append("found extern mm/addmm")
        notes.append("found a separate relu pointwise kernel")
    elif separate_relu:
        verdict = "separate_relu_after_matmul"
        notes.append("found a separate relu kernel")
    else:
        verdict = "unknown_check_log"
        notes.append("did not find enough markers for a confident classification")
    return verdict, notes


def launch_with_output_code(args: argparse.Namespace) -> None:
    log_path = args.log_path
    if log_path is None:
        log_path = Path(
            f"inductor_mm_relu_output_code_m{args.m}_n{args.n}_k{args.k}_{args.dtype}.log"
        )

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--dtype",
        args.dtype,
        "--mode",
        args.mode,
        "--float32-matmul-precision",
        args.float32_matmul_precision,
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--seed",
        str(args.seed),
    ]
    if args.dynamic:
        cmd.append("--dynamic")
    else:
        cmd.append("--no-dynamic")
    if args.force_exit:
        cmd.append("--force-exit")

    env = os.environ.copy()
    env["TORCH_LOGS"] = "output_code"
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_mm_relu_poc")

    result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    log_text = result.stdout + result.stderr
    log_path.write_text(log_text)

    print(result.stdout, end="")
    verdict, notes = analyze_output_code(log_text)
    print(f"output_code_log={log_path}")
    print(f"verdict={verdict}")
    for note in notes:
        print(f"- {note}")

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--mode", default="max-autotune")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--float32-matmul-precision", default="high")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()

    if args.child:
        run_child(args)
    else:
        launch_with_output_code(args)


if __name__ == "__main__":
    main()
