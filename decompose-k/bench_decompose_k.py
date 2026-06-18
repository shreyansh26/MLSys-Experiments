"""Run Decompose-K matmul and ReLU epilogue benchmarks."""

import argparse
import csv
import importlib
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable

import torch
import triton

# from kernels.decompose_k_triton_kernel import (
#     KernelConfig,
#     candidate_configs,
#     decompose_k_matmul_out,
#     decompose_k_relu_out,
#     inductor_like_splits,
# )
from kernels.decompose_k_triton_kernel_optimized import (
    KernelConfig,
    candidate_configs,
    decompose_k_matmul_out,
    decompose_k_relu_out,
    inductor_like_splits,
)

from custom_op_autotune_relu_dispatch import (
    custom_mm,
    custom_relu_mm,
    register_mm_static_autotune,
    register_mm_relu_static_autotune,
)


DEFAULT_MNS = [16, 32, 48, 64]
DEFAULT_KS = [8192, 12288, 16384, 20480, 24576, 28672, 32768]
_CUSTOM_MM_REGISTERED = False
_CUSTOM_MM_RELU_REGISTERED = False
CUTEDSL_MODULES = {
    "epilogue-bf16": "kernels.cutedsl_splitk_gemm_candidate",
    "matmul-bf16": "kernels.cutedsl_splitk_gemm_candidate",
    "matmul-fp16": "kernels.cutedsl_splitk_gemm_candidate",
    "matmul-fp32": "kernels.cutedsl_fp32_shared16_candidate",
}


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


@dataclass(frozen=True)
class CustomOpWinner:
    autotune_name: str
    impl: str
    k_splits: int | None
    choice_name: str


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
    "matmul-fp16": Suite(
        name="matmul-fp16",
        title="Plain Matmul, FP16",
        dtype=torch.float16,
        epilogue=False,
        csv_name="plain_matmul_fp16.csv",
        plot_prefix="plain_matmul_fp16",
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


def ensure_custom_mm_registered() -> None:
    global _CUSTOM_MM_REGISTERED
    if not _CUSTOM_MM_REGISTERED:
        register_mm_static_autotune()
        _CUSTOM_MM_REGISTERED = True


def ensure_custom_mm_relu_registered() -> None:
    global _CUSTOM_MM_RELU_REGISTERED
    if not _CUSTOM_MM_RELU_REGISTERED:
        register_mm_relu_static_autotune()
        _CUSTOM_MM_RELU_REGISTERED = True


def custom_op_winner_from_choice(name: str, choice: object) -> CustomOpWinner:
    decomposition = getattr(choice, "decomposition", None)
    kwargs = getattr(choice, "decomposition_kwargs", {}) or {}
    choice_name = getattr(choice, "name", type(choice).__name__)
    impl = decomposition.__name__ if decomposition is not None else "fallback"
    return CustomOpWinner(
        autotune_name=name,
        impl=impl,
        k_splits=kwargs.get("k_splits"),
        choice_name=str(choice_name),
    )


def compile_custom_op_with_winner(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    compile_mode: str,
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    torch.Tensor,
    CustomOpWinner,
]:
    import torch._inductor.kernel.custom_op as custom_op_kernel

    original_autotune_select = custom_op_kernel.autotune_select_algorithm
    winner: CustomOpWinner | None = None

    def capture_autotune_select(*args: object, **kwargs: object) -> tuple[object, object]:
        nonlocal winner
        selected_result, winning_choice = original_autotune_select(*args, **kwargs)
        name = str(kwargs.get("name", args[0] if args else "unknown"))
        if name.startswith("static_mm"):
            winner = custom_op_winner_from_choice(name, winning_choice)
        return selected_result, winning_choice

    # Inductor does not expose this winner on the compiled callable. Capture it
    # from the custom-op lowering path during the compile triggered by first use.
    custom_op_kernel.autotune_select_algorithm = capture_autotune_select
    try:
        compiled_target = torch.compile(fn, mode=compile_mode, dynamic=False)
        out = compiled_target(a, b)
        torch.cuda.synchronize()
    finally:
        custom_op_kernel.autotune_select_algorithm = original_autotune_select

    if winner is None:
        winner = CustomOpWinner(
            autotune_name="not_captured",
            impl="not_captured",
            k_splits=None,
            choice_name="not_captured",
        )

    return compiled_target, out, winner


def bench_ms(fn: Callable[[], torch.Tensor], warmup: int, rep: int) -> float:
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


@cache
def load_cutedsl_module(module_name: str):
    return importlib.import_module(module_name)


def cutedsl_module_for_suite(suite: Suite):
    module_name = CUTEDSL_MODULES.get(suite.name)
    if module_name is None:
        return None
    return load_cutedsl_module(module_name)


def cutedsl_partials_dtype(module, suite: Suite) -> torch.dtype:
    if hasattr(module, "partials_dtype"):
        return module.partials_dtype(suite.name, suite.dtype)
    return torch.float32


def best_cutedsl_plain_config(
    module,
    suite: Suite,
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
    splits = module.inductor_like_splits(m, n, k, split_limit)
    if not splits:
        raise ValueError(f"no valid CuteDSL split_k values for M={m}, N={n}, K={k}")

    best = (float("inf"), None)
    first_error = None
    partials_dtype = cutedsl_partials_dtype(module, suite)
    for config in module.candidate_configs(splits):
        c = torch.empty_like(ref)
        partials = torch.empty((config.split_k, m, n), device=a.device, dtype=partials_dtype)
        try:
            module.decompose_k_matmul_out(a, b, c, partials, config)
            torch.cuda.synchronize()
            torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)
            ms = bench_ms(
                lambda: module.decompose_k_matmul_out(a, b, c, partials, config),
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
            f"no CuteDSL configs completed for M={m}, N={n}, K={k}; "
            f"first failure: {first_error}"
        )
    return best


def best_cutedsl_epilogue_config(
    module,
    suite: Suite,
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
    splits = module.inductor_like_splits(m, n, k, split_limit)
    if not splits:
        raise ValueError(f"no valid CuteDSL split_k values for M={m}, N={n}, K={k}")

    best = (float("inf"), float("inf"), None)
    first_error = None
    partials_dtype = cutedsl_partials_dtype(module, suite)
    for config in module.candidate_configs(splits):
        c = torch.empty_like(ref)
        partials = torch.empty((config.split_k, m, n), device=a.device, dtype=partials_dtype)
        try:
            module.decompose_k_relu_out(a, b, c, partials, config, fuse_relu=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)
            fused_ms = bench_ms(
                lambda: module.decompose_k_relu_out(
                    a, b, c, partials, config, fuse_relu=True
                ),
                warmup,
                rep,
            )
            unfused_ms = bench_ms(
                lambda: module.decompose_k_matmul_out(a, b, c, partials, config).relu_(),
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
            f"no CuteDSL configs completed for M={m}, N={n}, K={k}; "
            f"first failure: {first_error}"
        )
    return best


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


def line_specs(suite: Suite, include_cutedsl: bool) -> list[tuple[str, str]]:
    if suite.epilogue:
        specs = [
            ("eager_ms", "torch.mm + relu"),
            ("compiled_ms", "compiled torch.mm + relu"),
            ("custom_op_mm_relu_ms", "custom op autotuned mm+relu"),
            ("decompose_k_unfused_ms", "decomposeK + relu"),
            ("decompose_k_fused_ms", "decomposeK fused relu"),
        ]
        if include_cutedsl and suite.name in CUTEDSL_MODULES:
            specs.extend(
                [
                    ("cutedsl_unfused_ms", "CuteDSL + relu"),
                    ("cutedsl_fused_ms", "CuteDSL fused relu"),
                ]
            )
        return specs

    specs = [
        ("eager_ms", "torch.mm"),
        ("compiled_ms", "compiled torch.mm"),
        ("custom_op_mm_ms", "custom op autotuned mm"),
        ("decompose_k_ms", "decomposeK"),
    ]
    if include_cutedsl and suite.name in CUTEDSL_MODULES:
        specs.append(("cutedsl_ms", "CuteDSL"))
    return specs


def plot_one(
    rows: list[dict[str, object]],
    suite: Suite,
    out_dir: Path,
    name_suffix: str,
    title_suffix: str,
    include_cutedsl: bool,
) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: (int(row["m"]), int(row["k"])))
    labels = [f"({row['m']}, {row['k']})" for row in rows]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(18, 6) if not title_suffix else (12, 5))
    for key, label in line_specs(suite, include_cutedsl):
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


def plot_results(
    rows: list[dict[str, object]],
    suite: Suite,
    out_dir: Path,
    include_cutedsl: bool,
) -> None:
    plot_one(rows, suite, out_dir, "_overall_grid", ", Overall Comparison Grid", include_cutedsl)

    for mn in sorted({int(row["m"]) for row in rows}):
        subset = [row for row in rows if int(row["m"]) == mn]
        plot_one(subset, suite, out_dir, f"_mn{mn}", f", M=N={mn}", include_cutedsl)


def cutedsl_config_fieldnames() -> list[str]:
    return [
        "cutedsl_split_k",
        "cutedsl_block_m",
        "cutedsl_block_n",
        "cutedsl_block_k",
        "cutedsl_num_warps",
        "cutedsl_num_stages",
    ]


def csv_fieldnames(suite: Suite, include_cutedsl: bool) -> list[str]:
    base = ["suite", "m", "n", "k", "dtype"]
    custom_config = [
        "custom_op_autotune_name",
        "custom_op_best_impl",
        "custom_op_k_splits",
        "custom_op_choice_name",
    ]
    standalone_config = [
        "standalone_split_k",
        "standalone_block_m",
        "standalone_block_n",
        "standalone_block_k",
    ]
    has_cutedsl = include_cutedsl and suite.name in CUTEDSL_MODULES
    if suite.epilogue:
        timings = [
            *base,
            "eager_ms",
            "compiled_ms",
            "custom_op_mm_relu_ms",
            "decompose_k_unfused_ms",
            "decompose_k_fused_ms",
        ]
        if has_cutedsl:
            timings.extend(["cutedsl_unfused_ms", "cutedsl_fused_ms"])

        speedups = [
            "custom_op_speedup_vs_eager",
            "custom_op_speedup_vs_compiled",
            "decompose_k_fused_speedup_vs_eager",
            "decompose_k_fused_speedup_vs_compiled",
            "decompose_k_fused_vs_unfused_speedup",
        ]
        if has_cutedsl:
            speedups.extend(
                [
                    "cutedsl_fused_speedup_vs_eager",
                    "cutedsl_fused_speedup_vs_compiled",
                    "cutedsl_fused_vs_unfused_speedup",
                    "decompose_k_fused_over_cutedsl_fused_speedup",
                ]
            )

        return [
            *timings,
            *speedups,
            *custom_config,
            *standalone_config,
            *(cutedsl_config_fieldnames() if has_cutedsl else []),
        ]

    timings = [
        *base,
        "eager_ms",
        "compiled_ms",
        "custom_op_mm_ms",
        "decompose_k_ms",
    ]
    if has_cutedsl:
        timings.append("cutedsl_ms")

    speedups = [
        "custom_op_speedup_vs_eager",
        "custom_op_speedup_vs_compiled",
        "decompose_k_speedup_vs_eager",
        "decompose_k_speedup_vs_compiled",
    ]
    if has_cutedsl:
        speedups.extend(
            [
                "cutedsl_speedup_vs_eager",
                "cutedsl_speedup_vs_compiled",
                "decompose_k_over_cutedsl_speedup",
            ]
        )

    return [
        *timings,
        *speedups,
        *custom_config,
        *standalone_config,
        *(cutedsl_config_fieldnames() if has_cutedsl else []),
    ]


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
    include_cutedsl: bool,
) -> None:
    target = mm_relu if suite.epilogue else mm_only
    torch.set_float32_matmul_precision("highest" if suite.dtype == torch.float32 else "high")
    if suite.epilogue:
        ensure_custom_mm_relu_registered()
    else:
        ensure_custom_mm_registered()
    rtol = suite.rtol if rtol is None else rtol
    atol = suite.atol if atol is None else atol
    cutedsl_module = cutedsl_module_for_suite(suite) if include_cutedsl else None

    print(f"\n== {suite.name} ==", flush=True)
    print(f"dtype={suite.dtype} compile_mode={compile_mode}", flush=True)
    print(f"assert_close rtol={rtol} atol={atol}", flush=True)
    if cutedsl_module is not None:
        print(f"cutedsl_module={CUTEDSL_MODULES[suite.name]}", flush=True)

    rows: list[dict[str, object]] = []
    for mn in mns:
        for k in ks:
            a = torch.randn((mn, k), device="cuda", dtype=suite.dtype)
            b = torch.randn((k, mn), device="cuda", dtype=suite.dtype)
            ref = target(a, b)
            torch.cuda.synchronize()

            # Each grid point is an exact-shape benchmark. Reset Dynamo so the
            # sweep does not hit the per-function recompile limit and fall back
            # to slower execution after several K values.
            torch._dynamo.reset()
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
                custom_compiled_target, custom_out, custom_winner = (
                    compile_custom_op_with_winner(
                        custom_relu_mm,
                        a,
                        b,
                        compile_mode,
                    )
                )
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
                        "decompose_k_fused_speedup_vs_eager": eager_ms / fused_ms,
                        "decompose_k_fused_speedup_vs_compiled": compiled_ms / fused_ms,
                        "decompose_k_fused_vs_unfused_speedup": unfused_ms / fused_ms,
                    }
                )
                cutedsl_metric = ""
                cutedsl_config = None
                if cutedsl_module is not None:
                    cutedsl_fused_ms, cutedsl_unfused_ms, cutedsl_config = (
                        best_cutedsl_epilogue_config(
                            cutedsl_module,
                            suite,
                            a,
                            b,
                            ref,
                            split_limit,
                            warmup,
                            rep,
                            rtol,
                            atol,
                        )
                    )
                    row.update(
                        {
                            "cutedsl_unfused_ms": cutedsl_unfused_ms,
                            "cutedsl_fused_ms": cutedsl_fused_ms,
                            "cutedsl_fused_speedup_vs_eager": eager_ms / cutedsl_fused_ms,
                            "cutedsl_fused_speedup_vs_compiled": (
                                compiled_ms / cutedsl_fused_ms
                            ),
                            "cutedsl_fused_vs_unfused_speedup": (
                                cutedsl_unfused_ms / cutedsl_fused_ms
                            ),
                            "decompose_k_fused_over_cutedsl_fused_speedup": (
                                fused_ms / cutedsl_fused_ms
                            ),
                        }
                    )
                    cutedsl_metric = (
                        f" cutedsl_unfused={cutedsl_unfused_ms:.4f}ms "
                        f"cutedsl_fused={cutedsl_fused_ms:.4f}ms "
                        f"triton/cutedsl={fused_ms / cutedsl_fused_ms:.2f}x"
                    )
                metric = (
                    f"custom_op={custom_op_ms:.4f}ms "
                    f"unfused={unfused_ms:.4f}ms fused={fused_ms:.4f}ms "
                    f"speedup_vs_compiled={compiled_ms / fused_ms:.2f}x"
                    f"{cutedsl_metric}"
                )
            else:
                custom_compiled_target, custom_out, custom_winner = (
                    compile_custom_op_with_winner(
                        custom_mm,
                        a,
                        b,
                        compile_mode,
                    )
                )
                torch.testing.assert_close(custom_out, ref, rtol=rtol, atol=atol)
                custom_op_ms = bench_ms(
                    lambda: custom_compiled_target(a, b),
                    warmup,
                    rep,
                )
                decompose_ms, config = best_decompose_k_plain_config(
                    a, b, ref, split_limit, warmup, rep, rtol, atol
                )
                row.update(
                    {
                        "custom_op_mm_ms": custom_op_ms,
                        "custom_op_speedup_vs_eager": eager_ms / custom_op_ms,
                        "custom_op_speedup_vs_compiled": compiled_ms / custom_op_ms,
                        "decompose_k_ms": decompose_ms,
                        "decompose_k_speedup_vs_eager": eager_ms / decompose_ms,
                        "decompose_k_speedup_vs_compiled": compiled_ms / decompose_ms,
                    }
                )
                cutedsl_metric = ""
                cutedsl_config = None
                if cutedsl_module is not None:
                    cutedsl_ms, cutedsl_config = best_cutedsl_plain_config(
                        cutedsl_module,
                        suite,
                        a,
                        b,
                        ref,
                        split_limit,
                        warmup,
                        rep,
                        rtol,
                        atol,
                    )
                    row.update(
                        {
                            "cutedsl_ms": cutedsl_ms,
                            "cutedsl_speedup_vs_eager": eager_ms / cutedsl_ms,
                            "cutedsl_speedup_vs_compiled": compiled_ms / cutedsl_ms,
                            "decompose_k_over_cutedsl_speedup": decompose_ms / cutedsl_ms,
                        }
                    )
                    cutedsl_metric = (
                        f" cutedsl={cutedsl_ms:.4f}ms "
                        f"triton/cutedsl={decompose_ms / cutedsl_ms:.2f}x"
                    )
                metric = (
                    f"custom_op={custom_op_ms:.4f}ms "
                    f"decomposeK={decompose_ms:.4f}ms "
                    f"speedup_vs_compiled={compiled_ms / decompose_ms:.2f}x"
                    f"{cutedsl_metric}"
                )

            row.update(
                {
                    "custom_op_autotune_name": custom_winner.autotune_name,
                    "custom_op_best_impl": custom_winner.impl,
                    "custom_op_k_splits": (
                        "" if custom_winner.k_splits is None else custom_winner.k_splits
                    ),
                    "custom_op_choice_name": custom_winner.choice_name,
                    "standalone_split_k": config.split_k,
                    "standalone_block_m": config.block_m,
                    "standalone_block_n": config.block_n,
                    "standalone_block_k": config.block_k,
                }
            )
            if cutedsl_config is not None:
                row.update(
                    {
                        "cutedsl_split_k": cutedsl_config.split_k,
                        "cutedsl_block_m": cutedsl_config.block_m,
                        "cutedsl_block_n": cutedsl_config.block_n,
                        "cutedsl_block_k": cutedsl_config.block_k,
                        "cutedsl_num_warps": cutedsl_config.num_warps,
                        "cutedsl_num_stages": cutedsl_config.num_stages,
                    }
                )
            rows.append(row)
            cutedsl_config_text = ""
            if cutedsl_config is not None:
                cutedsl_config_text = (
                    f" cutedsl=split{cutedsl_config.split_k}/"
                    f"bm{cutedsl_config.block_m}/bn{cutedsl_config.block_n}/"
                    f"bk{cutedsl_config.block_k}/w{cutedsl_config.num_warps}/"
                    f"s{cutedsl_config.num_stages}"
                )
            print(
                f"(M=N={mn}, K={k}) eager={eager_ms:.4f}ms "
                f"compiled={compiled_ms:.4f}ms {metric} "
                f"custom={custom_winner.impl}/split{custom_winner.k_splits} "
                f"standalone=split{config.split_k}/bm{config.block_m}/"
                f"bn{config.block_n}/bk{config.block_k}"
                f"{cutedsl_config_text}",
                flush=True,
            )

    csv_path = out_dir / suite.csv_name
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fieldnames(suite, cutedsl_module is not None),
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_results(rows, suite, out_dir, cutedsl_module is not None)
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
    parser.add_argument(
        "--no-cutedsl",
        action="store_true",
        help="skip CuteDSL kernels and keep the historical benchmark columns/plots",
    )
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
            include_cutedsl=not args.no_cutedsl,
        )

    if os.environ.get("DECOMPOSE_K_FORCE_EXIT", "1") == "1":
        os._exit(0)


if __name__ == "__main__":
    main()
