"""Correctness checks and Triton-tutorial-style GEMM benchmark plots."""

from __future__ import annotations

import argparse
from pathlib import Path

# Import this first so the TLX compiler plugin is registered before Triton use.
from tlx_plugin import load_tlx

load_tlx()

import torch
import triton

from reference import supports_triton_warp_specialization, torch_matmul
from triton_tma_matmul import (
    matmul as triton_tma_matmul,
)
from triton_tma_matmul import (
    warp_specialized_matmul as triton_tma_ws_matmul,
)
from triton_tma_persistent_matmul import (
    matmul as triton_tma_persistent_matmul,
)
from triton_tma_persistent_matmul import (
    warp_specialized_matmul as triton_tma_persistent_ws_matmul,
)

IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
if IS_BLACKWELL:
    from blackwell.gluon_matmul import (
        matmul as gluon_matmul,
        matmul_one_cta as gluon_matmul_one_cta,
        matmul_two_cta as gluon_matmul_two_cta,
    )
    from blackwell.tlx_matmul import matmul as tlx_matmul
else:
    from gluon_matmul import matmul as gluon_matmul
    from tlx_matmul import matmul as tlx_matmul


QUICK_SHAPES = [
    (129, 127, 131),  # exercises every boundary mask and TLX padding
    (512, 512, 512),
    (1024, 1024, 1024),
]

FULL_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (1024, 4096, 4096),  # wide output
    (4096, 1024, 4096),  # tall output
    (4096, 11008, 4096),  # Llama-style up projection
    (4096, 4096, 11008),  # Llama-style down projection
]

TRITON_WS = supports_triton_warp_specialization()
PROVIDERS = {
    "pytorch": torch_matmul,
    "triton-tma": triton_tma_ws_matmul if TRITON_WS else triton_tma_matmul,
    "triton-persistent": (
        triton_tma_persistent_ws_matmul if TRITON_WS else triton_tma_persistent_matmul
    ),
    "tlx": tlx_matmul,
    "gluon": gluon_matmul,
}
if IS_BLACKWELL:
    PROVIDERS.update(
        {
            "gluon-1cta": gluon_matmul_one_cta,
            "gluon-2cta": gluon_matmul_two_cta,
        }
    )

DEFAULT_PROVIDERS = [
    "pytorch",
    "triton-tma",
    "triton-persistent",
    "tlx",
    "gluon",
]


def check_correctness(providers: list[str] | None = None) -> None:
    providers = providers or list(PROVIDERS)
    torch.manual_seed(0)
    shapes = [(1, 1, 1), (33, 65, 17), (129, 127, 131), (256, 384, 192)]
    for dtype in (torch.float16, torch.bfloat16):
        for m, n, k in shapes:
            a = torch.randn((m, k), device="cuda", dtype=dtype)
            b = torch.randn((k, n), device="cuda", dtype=dtype)
            expected = torch_matmul(a, b)
            for name in providers:
                implementation = PROVIDERS[name]
                actual = implementation(a, b)
                torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)
                print(f"PASS {name:18s} {dtype!s:14s} M={m:4d} N={n:4d} K={k:4d}")


def _measure(m: int, n: int, k: int, provider: str, dtype: torch.dtype):
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    implementation = PROVIDERS[provider]
    return triton.testing.do_bench(
        lambda: implementation(a, b), quantiles=[0.5, 0.2, 0.8]
    )


def make_report(
    shapes: list[tuple[int, int, int]],
    dtype: torch.dtype,
    metric: str,
    providers: list[str],
):
    dtype_name = str(dtype).removeprefix("torch.")
    ylabel = "TFLOP/s" if metric == "tflops" else "Latency (ms)"
    shape_labels = [f"{m}x{n}x{k}" for m, n, k in shapes]

    triton_mode = "compiler WS" if TRITON_WS else "SM90 pipeline"
    custom_mode = "TCGen5/TMEM WS" if IS_BLACKWELL else "WGMMA WS"
    provider_names = {
        "pytorch": "PyTorch (cuBLAS)",
        "triton-tma": f"Triton TMA tiled ({triton_mode})",
        "triton-persistent": f"Triton TMA persistent ({triton_mode})",
        "tlx": f"TLX persistent {custom_mode}",
        "gluon": f"Gluon auto {custom_mode}",
        "gluon-1cta": "Gluon one-CTA TCGen5/TMEM WS",
        "gluon-2cta": "Gluon two-CTA multicast TCGen5/TMEM WS",
    }
    provider_styles = {
        "pytorch": ("green", "-"),
        "triton-tma": ("blue", "-"),
        "triton-persistent": ("cyan", "-"),
        "tlx": ("red", "-"),
        "gluon": ("orange", "-"),
        "gluon-1cta": ("purple", "-"),
        "gluon-2cta": ("brown", "-"),
    }

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["shape"],
            x_vals=shape_labels,
            line_arg="provider",
            line_vals=providers,
            line_names=[provider_names[name] for name in providers],
            styles=[provider_styles[name] for name in providers],
            ylabel=ylabel,
            plot_name=f"matmul-{dtype_name}-{metric}",
            args={},
        )
    )
    def benchmark(shape, provider):
        M, N, K = (int(value) for value in shape.split("x"))
        median_ms, low_ms, high_ms = _measure(M, N, K, provider, dtype)
        if metric == "latency":
            return median_ms, low_ms, high_ms

        def tflops(ms):
            return 2.0 * M * N * K * 1e-12 / (ms * 1e-3)

        # Faster time is the upper throughput error bar.
        return tflops(median_ms), tflops(high_ms), tflops(low_ms)

    return benchmark


def save_readable_plot(output: Path, dtype: torch.dtype, metric: str) -> None:
    """Turn perf_report's CSV into a readable discrete-shape comparison."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    dtype_name = str(dtype).removeprefix("torch.")
    csv_path = output / f"matmul-{dtype_name}-{metric}.csv"
    frame = pd.read_csv(csv_path)
    series = list(frame.columns[1:])
    colors = [
        "#3B6FB6",
        "#D58B18",
        "#2A9D8F",
        "#8E5BB7",
        "#737373",
        "#B56576",
        "#6D597A",
    ]
    hatches = ["", "//", "\\\\", "..", "xx", "++", "oo"]
    y = np.arange(len(frame))
    height = 0.15

    fig, ax = plt.subplots(figsize=(10, max(5.5, len(frame) * 0.72)))
    for index, (column, color, hatch) in enumerate(
        zip(series, colors[: len(series)], hatches[: len(series)], strict=True)
    ):
        offset = (index - (len(series) - 1) / 2) * height
        ax.barh(
            y + offset,
            frame[column],
            height,
            label=column.split(" (")[0],
            color=color,
            hatch=hatch,
            edgecolor="#333333",
            linewidth=0.5,
        )

    unit = (
        "TFLOP/s (higher is better)" if metric == "tflops" else "ms (lower is better)"
    )
    title_metric = "Throughput" if metric == "tflops" else "Latency"
    ax.set_title(
        f"{dtype_name.upper()} matrix multiplication {title_metric}",
        loc="left",
        pad=30,
    )
    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CUDA GPU"
    ax.text(
        0,
        1.01,
        f"{gpu_name}; median GPU time from triton.testing.do_bench; {unit}",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9,
    )
    ax.set_xlabel(unit)
    ax.set_ylabel("Shape (M × N × K)")
    ax.set_yticks(y, frame["shape"])
    ax.invert_yaxis()
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(
        output / f"matmul-{dtype_name}-{metric}-comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("quick", "full"), default="full")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--provider",
        choices=("all", "gluon-compare", *PROVIDERS),
        default="all",
        help="benchmark canonical providers, both Gluon CTA variants, or one implementation",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--skip-check", action="store_true", help="skip correctness checks"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    if args.provider == "all":
        providers = DEFAULT_PROVIDERS
    elif args.provider == "gluon-compare":
        if not IS_BLACKWELL:
            raise RuntimeError("Gluon CTA comparison requires Blackwell (SM100+)")
        providers = ["gluon", "gluon-1cta", "gluon-2cta"]
    else:
        providers = [args.provider]
    if not args.skip_check:
        check_correctness(providers)

    shapes = QUICK_SHAPES if args.suite == "quick" else FULL_SHAPES
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    args.output.mkdir(parents=True, exist_ok=True)
    for metric in ("tflops", "latency"):
        report = make_report(shapes, dtype, metric, providers)
        report.run(save_path=str(args.output), show_plots=False, print_data=True)
        save_readable_plot(args.output, dtype, metric)
    print(f"Saved CSVs and plots to {args.output.resolve()}")


if __name__ == "__main__":
    main()
