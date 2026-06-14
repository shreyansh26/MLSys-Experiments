"""Run Decompose-K matmul and ReLU epilogue benchmarks."""

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import triton

from decompose_k_kernel import (
    KernelConfig,
    candidate_configs,
    decompose_k_matmul_out,
    decompose_k_relu_out,
    inductor_like_splits,
)
from custom_op_autotune_relu_dispatch import (
    custom_relu_mm,
    register_mm_relu_static_autotune,
)


DEFAULT_MNS = [16, 32, 48, 64]
DEFAULT_KS = [8192, 12288, 16384, 20480, 24576, 28672, 32768]
_CUSTOM_MM_RELU_REGISTERED = False


@dataclass(frozen=True)
class Suite:
    name: str
    title: str
    dtype: torch.dtype
    epilogue: bool
    csv_name: str
    plot_prefix: str
    rtol: float
    atol: float


SUITES = {
    "epilogue-bf16": Suite(
        name="epilogue-bf16",
        title="ReLU Epilogue Fusion, BF16",
        dtype=torch.bfloat16,
        epilogue=True,
        csv_name="epilogue_relu_bf16.csv",
        plot_prefix="epilogue_relu_bf16",
        rtol=2e-2,
        atol=1e-2,
    ),
    "matmul-bf16": Suite(
        name="matmul-bf16",
        title="Plain Matmul, BF16",
        dtype=torch.bfloat16,
        epilogue=False,
        csv_name="plain_matmul_bf16.csv",
        plot_prefix="plain_matmul_bf16",
        rtol=2e-2,
        atol=1e-2,
    ),
    "matmul-fp32": Suite(
        name="matmul-fp32",
        title="Plain Matmul, FP32",
        dtype=torch.float32,
        epilogue=False,
        csv_name="plain_matmul_fp32.csv",
        plot_prefix="plain_matmul_fp32",
        rtol=1e-4,
        atol=1e-3,
    ),
}


def parse_csv_ints(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def parse_suites(text: str) -> list[Suite]:
    if text == "all":
        return list(SUITES.values())

    suites = []
    for name in text.split(","):
        if not name:
            continue
        try:
            suites.append(SUITES[name])
        except KeyError as exc:
            valid = ", ".join(["all", *SUITES])
            raise argparse.ArgumentTypeError(f"unknown suite {name!r}; valid: {valid}") from exc
    if not suites:
        raise argparse.ArgumentTypeError("expected at least one benchmark suite")
    return suites


def mm_only(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


def mm_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.mm(a, b))


def ensure_custom_mm_relu_registered() -> None:
    global _CUSTOM_MM_RELU_REGISTERED
    if not _CUSTOM_MM_RELU_REGISTERED:
        register_mm_relu_static_autotune()
        _CUSTOM_MM_RELU_REGISTERED = True


def bench_ms(fn: Callable[[], torch.Tensor], warmup: int, rep: int) -> float:
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def best_decompose_k_plain_config(
    a: torch.Tensor,
    b: torch.Tensor,
    ref: torch.Tensor,
    split_limit: int,
    warmup: int,
    rep: int,
    rtol: float,
    atol: float,
) -> tuple[float, KernelConfig]:
    m, k = a.shape
    n = b.shape[1]
    splits = inductor_like_splits(m, n, k, split_limit)
    if not splits:
        raise ValueError(f"no valid split_k values for M={m}, N={n}, K={k}")

    best = (float("inf"), None)
    first_error = None
    for config in candidate_configs(splits):
        c = torch.empty_like(ref)
        partials = torch.empty((config.split_k, m, n), device=a.device, dtype=torch.float32)
        try:
            decompose_k_matmul_out(a, b, c, partials, config)
            torch.cuda.synchronize()
            torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)
            ms = bench_ms(
                lambda: decompose_k_matmul_out(a, b, c, partials, config),
                warmup,
                rep,
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue

        if ms < best[0]:
            best = (ms, config)

    if best[1] is None:
        raise RuntimeError(
            f"no Decompose-K configs completed for M={m}, N={n}, K={k}; "
            f"first failure: {first_error}"
        )
    return best


def best_decompose_k_epilogue_config(
    a: torch.Tensor,
    b: torch.Tensor,
    ref: torch.Tensor,
    split_limit: int,
    warmup: int,
    rep: int,
    rtol: float,
    atol: float,
) -> tuple[float, float, KernelConfig]:
    m, k = a.shape
    n = b.shape[1]
    splits = inductor_like_splits(m, n, k, split_limit)
    if not splits:
        raise ValueError(f"no valid split_k values for M={m}, N={n}, K={k}")

    best = (float("inf"), float("inf"), None)
    first_error = None
    for config in candidate_configs(splits):
        c = torch.empty_like(ref)
        partials = torch.empty((config.split_k, m, n), device=a.device, dtype=torch.float32)
        try:
            decompose_k_relu_out(a, b, c, partials, config, fuse_relu=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)
            fused_ms = bench_ms(
                lambda: decompose_k_relu_out(a, b, c, partials, config, fuse_relu=True),
                warmup,
                rep,
            )
            unfused_ms = bench_ms(
                lambda: decompose_k_matmul_out(a, b, c, partials, config).relu_(),
                warmup,
                rep,
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue

        if fused_ms < best[0]:
            best = (fused_ms, unfused_ms, config)

    if best[2] is None:
        raise RuntimeError(
            f"no Decompose-K configs completed for M={m}, N={n}, K={k}; "
            f"first failure: {first_error}"
        )
    return best


def line_specs(suite: Suite) -> list[tuple[str, str]]:
    if suite.epilogue:
        return [
            ("eager_ms", "torch.mm + relu"),
            ("compiled_ms", "compiled torch.mm + relu"),
            ("custom_op_mm_relu_ms", "custom op autotuned mm+relu"),
            ("decompose_k_unfused_ms", "decomposeK + relu"),
            ("decompose_k_fused_ms", "decomposeK fused relu"),
        ]
    return [
        ("eager_ms", "torch.mm"),
        ("compiled_ms", "compiled torch.mm"),
        ("decompose_k_ms", "decomposeK"),
    ]


def plot_one(
    rows: list[dict[str, object]],
    suite: Suite,
    out_dir: Path,
    name_suffix: str,
    title_suffix: str,
) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: (int(row["m"]), int(row["k"])))
    labels = [f"({row['m']}, {row['k']})" for row in rows]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(18, 6) if not title_suffix else (12, 5))
    for key, label in line_specs(suite):
        ax.plot(x, [float(row[key]) for row in rows], marker="o", label=label)

    if not title_suffix:
        group_start = 0
        for mn in sorted({int(row["m"]) for row in rows}):
            count = sum(1 for row in rows if int(row["m"]) == mn)
            if group_start:
                ax.axvline(group_start - 0.5, color="0.75", linewidth=1)
            ax.text(
                group_start + (count - 1) / 2,
                0.985,
                f"M=N={mn}",
                ha="center",
                va="top",
                color="0.25",
                fontsize=10,
                transform=ax.get_xaxis_transform(),
            )
            group_start += count

    ax.set_title(f"{suite.title}{title_suffix}")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("(M/N, K)")
    ax.set_xticks(x, labels, rotation=60 if not title_suffix else 45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{suite.plot_prefix}{name_suffix}.png", dpi=180)
    plt.close(fig)


def plot_results(rows: list[dict[str, object]], suite: Suite, out_dir: Path) -> None:
    plot_one(rows, suite, out_dir, "_overall_grid", ", Overall Comparison Grid")

    for mn in sorted({int(row["m"]) for row in rows}):
        subset = [row for row in rows if int(row["m"]) == mn]
        plot_one(subset, suite, out_dir, f"_mn{mn}", f", M=N={mn}")


def run_suite(
    suite: Suite,
    mns: list[int],
    ks: list[int],
    split_limit: int,
    warmup: int,
    rep: int,
    compile_mode: str,
    rtol: float | None,
    atol: float | None,
    out_dir: Path,
) -> None:
    target = mm_relu if suite.epilogue else mm_only
    torch.set_float32_matmul_precision("highest" if suite.dtype == torch.float32 else "high")
    if suite.epilogue:
        ensure_custom_mm_relu_registered()
    rtol = suite.rtol if rtol is None else rtol
    atol = suite.atol if atol is None else atol

    print(f"\n== {suite.name} ==", flush=True)
    print(f"dtype={suite.dtype} compile_mode={compile_mode}", flush=True)
    print(f"assert_close rtol={rtol} atol={atol}", flush=True)

    rows: list[dict[str, object]] = []
    for mn in mns:
        for k in ks:
            a = torch.randn((mn, k), device="cuda", dtype=suite.dtype)
            b = torch.randn((k, mn), device="cuda", dtype=suite.dtype)
            ref = target(a, b)
            torch.cuda.synchronize()

            compiled_target = torch.compile(target, mode=compile_mode, dynamic=False)
            compiled_out = compiled_target(a, b)
            torch.cuda.synchronize()
            torch.testing.assert_close(compiled_out, ref, rtol=rtol, atol=atol)

            eager_ms = bench_ms(lambda: target(a, b), warmup, rep)
            compiled_ms = bench_ms(lambda: compiled_target(a, b), warmup, rep)

            row: dict[str, object] = {
                "suite": suite.name,
                "m": mn,
                "n": mn,
                "k": k,
                "dtype": str(suite.dtype).removeprefix("torch."),
                "eager_ms": eager_ms,
                "compiled_ms": compiled_ms,
            }
            if suite.epilogue:
                custom_compiled_target = torch.compile(
                    custom_relu_mm,
                    mode=compile_mode,
                    dynamic=False,
                )
                custom_out = custom_compiled_target(a, b)
                torch.cuda.synchronize()
                torch.testing.assert_close(custom_out, ref, rtol=rtol, atol=atol)
                custom_op_ms = bench_ms(
                    lambda: custom_compiled_target(a, b),
                    warmup,
                    rep,
                )
                fused_ms, unfused_ms, config = best_decompose_k_epilogue_config(
                    a, b, ref, split_limit, warmup, rep, rtol, atol
                )
                row.update(
                    {
                        "custom_op_mm_relu_ms": custom_op_ms,
                        "custom_op_speedup_vs_eager": eager_ms / custom_op_ms,
                        "custom_op_speedup_vs_compiled": compiled_ms / custom_op_ms,
                        "decompose_k_unfused_ms": unfused_ms,
                        "decompose_k_fused_ms": fused_ms,
                        "fused_speedup_vs_eager": eager_ms / fused_ms,
                        "fused_speedup_vs_compiled": compiled_ms / fused_ms,
                        "epilogue_fusion_speedup": unfused_ms / fused_ms,
                    }
                )
                metric = (
                    f"custom_op={custom_op_ms:.4f}ms "
                    f"unfused={unfused_ms:.4f}ms fused={fused_ms:.4f}ms "
                    f"speedup_vs_compiled={compiled_ms / fused_ms:.2f}x"
                )
            else:
                decompose_ms, config = best_decompose_k_plain_config(
                    a, b, ref, split_limit, warmup, rep, rtol, atol
                )
                row.update(
                    {
                        "decompose_k_ms": decompose_ms,
                        "decompose_k_speedup_vs_eager": eager_ms / decompose_ms,
                        "decompose_k_speedup_vs_compiled": compiled_ms / decompose_ms,
                    }
                )
                metric = (
                    f"decomposeK={decompose_ms:.4f}ms "
                    f"speedup_vs_compiled={compiled_ms / decompose_ms:.2f}x"
                )

            row.update(
                {
                    "split_k": config.split_k,
                    "block_m": config.block_m,
                    "block_n": config.block_n,
                    "block_k": config.block_k,
                }
            )
            rows.append(row)
            print(
                f"(M=N={mn}, K={k}) eager={eager_ms:.4f}ms "
                f"compiled={compiled_ms:.4f}ms {metric} "
                f"config=split{config.split_k}/bm{config.block_m}/bn{config.block_n}/bk{config.block_k}",
                flush=True,
            )

    csv_path = out_dir / suite.csv_name
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_results(rows, suite, out_dir)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote plots for {suite.name} under {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", type=parse_suites, default=list(SUITES.values()))
    parser.add_argument("--mns", type=parse_csv_ints, default=DEFAULT_MNS)
    parser.add_argument("--ks", type=parse_csv_ints, default=DEFAULT_KS)
    parser.add_argument("--split-limit", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--out-dir", type=Path, default=Path("bench_results"))
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    print(f"torch={torch.__version__} triton={triton.__version__}", flush=True)
    print(f"device={torch.cuda.get_device_name()}", flush=True)
    print(f"shapes=M/N:{args.mns} K:{args.ks}", flush=True)

    for suite in args.suites:
        run_suite(
            suite=suite,
            mns=args.mns,
            ks=args.ks,
            split_limit=args.split_limit,
            warmup=args.warmup,
            rep=args.rep,
            compile_mode=args.compile_mode,
            rtol=args.rtol,
            atol=args.atol,
            out_dir=args.out_dir,
        )

    if os.environ.get("DECOMPOSE_K_FORCE_EXIT", "1") == "1":
        os._exit(0)


if __name__ == "__main__":
    main()
