"""Correctness checks for TileLang SGMV shrink and expand kernels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import torch

from sgmv_expand_tileang import sgmv_expand
from sgmv_shrink_tilelang import sgmv_shrink


@dataclass
class TestConfig:
    batch_size: int
    tokens_per_batch: int
    f_in: int
    rank: int
    f_out: int
    num_loras: int
    num_layers: int
    layer_idx: int
    scale: float
    accumulate: bool


def _build_metadata(indices: torch.Tensor, num_loras: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive auxiliary metadata tensors required by the kernels."""
    token_indices_sorted = indices.argsort(stable=True)
    num_tokens_per_lora = torch.zeros(num_loras + 1, dtype=indices.dtype, device=indices.device)
    lora_token_start_loc = torch.zeros(num_loras + 2, dtype=indices.dtype, device=indices.device)
    active_lora_ids = torch.ones(num_loras + 1, dtype=indices.dtype, device=indices.device) * num_loras

    unique_ids, counts = torch.unique(indices, sorted=True, return_counts=True)
    active_lora_ids[: unique_ids.shape[0]] = unique_ids
    num_tokens_per_lora[: counts.shape[0]] = counts
    lora_token_start_loc[1 : 1 + counts.shape[0]] = torch.cumsum(counts, dim=0)

    return token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, active_lora_ids


def _run_single_test(cfg: TestConfig, device: torch.device, dtype: torch.dtype, rtol: float, atol: float) -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    total_tokens = cfg.batch_size * cfg.tokens_per_batch
    assert cfg.layer_idx < cfg.num_layers, "layer_idx must be within [0, num_layers)"

    X = torch.randn(total_tokens, cfg.f_in, device=device, dtype=dtype)
    Y_shrink = torch.empty(total_tokens, cfg.rank, device=device, dtype=dtype)

    # LoRA weights shaped [num_loras * num_layers, rank, f_in] for shrink
    W_shrink_full = torch.randn(cfg.num_loras * cfg.num_layers, cfg.rank, cfg.f_in, device=device, dtype=dtype)
    layer_selector = torch.arange(cfg.num_loras, device=device) * cfg.num_layers + cfg.layer_idx
    W_shrink = W_shrink_full[layer_selector]

    # Indices per original batch, repeat for inner tokens
    batch_indices = torch.randint(0, cfg.num_loras + 1, (cfg.batch_size,), device=device, dtype=torch.int32)
    if cfg.batch_size > 0:
        batch_indices[0] = cfg.num_loras  # force a no-LoRA row
    indices = batch_indices.repeat_interleave(cfg.tokens_per_batch)

    (
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
    ) = _build_metadata(indices, cfg.num_loras)

    # SHRINK -----------------------------------------------------------------
    sgmv_shrink(
        Y_shrink,
        X,
        W_shrink,
        indices,
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        cfg.num_loras,
        cfg.scale,
    )

    idx_long = indices.to(torch.long)
    gather_ids = idx_long.clone()
    gather_ids[gather_ids == cfg.num_loras] = 0  # sentinel rows reuse first LoRA for indexing
    W_sel = W_shrink[gather_ids]
    ref_shrink = torch.einsum("brk,bk->br", W_sel, X)
    ref_shrink[idx_long == cfg.num_loras] = 0
    ref_shrink.mul_(cfg.scale)

    try:
        torch.testing.assert_close(Y_shrink, ref_shrink, rtol=rtol, atol=atol)
    except AssertionError as exc:
        diff = (Y_shrink - ref_shrink).abs()
        max_idx = diff.argmax()
        print(f"shrink failure: max diff={diff.max().item()} at flat index {max_idx}")
        raise

    # EXPAND -----------------------------------------------------------------
    X_rank = torch.randn(total_tokens, cfg.rank, device=device, dtype=dtype)
    Y_expand = torch.randn(total_tokens, cfg.f_out, device=device, dtype=dtype)
    Y_expand_base = Y_expand.clone()

    W_expand_full = torch.randn(cfg.num_loras * cfg.num_layers, cfg.f_out, cfg.rank, device=device, dtype=dtype)
    W_expand = W_expand_full[layer_selector]

    sgmv_expand(
        Y_expand,
        X_rank,
        W_expand,
        indices,
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        cfg.num_loras,
        cfg.accumulate,
    )

    gather_ids = idx_long.clone()
    gather_ids[gather_ids == cfg.num_loras] = 0
    W_expand_sel = W_expand[gather_ids]
    ref_expand = torch.einsum("bfr,br->bf", W_expand_sel, X_rank)
    ref_expand[idx_long == cfg.num_loras] = 0
    if cfg.accumulate:
        ref_expand.add_(Y_expand_base)

    try:
        torch.testing.assert_close(Y_expand, ref_expand, rtol=rtol, atol=atol)
    except AssertionError as exc:
        diff = (Y_expand - ref_expand).abs()
        max_idx = int(diff.argmax().item())
        row = max_idx // Y_expand.shape[1]
        col = max_idx % Y_expand.shape[1]
        idx_long = indices.to(torch.long)
        print(f"expand failure: max diff={diff.max().item()} at row {row}, col {col}, lora_id={idx_long[row].item()}")
        print('Y row:', Y_expand[row])
        print('ref row:', ref_expand[row])
        print('base row:', Y_expand_base[row])
        print('indices:', indices.view(-1))
        raise

    # Also check accumulate=False branch explicitly when requested
    if cfg.accumulate:
        Y_expand_zero = torch.zeros_like(Y_expand_base)
        sgmv_expand(
            Y_expand_zero,
            X_rank,
            W_expand,
            indices,
            token_indices_sorted,
            num_tokens_per_lora,
            lora_token_start_loc,
            active_lora_ids,
            cfg.num_loras,
            False,
        )
        ref_expand_reset = torch.einsum("bfr,br->bf", W_expand_sel, X_rank)
        ref_expand_reset[idx_long == cfg.num_loras] = 0
        torch.testing.assert_close(Y_expand_zero, ref_expand_reset, rtol=rtol, atol=atol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="Execution device (default: cuda)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Torch dtype for the test tensors")
    parser.add_argument("--rtol", type=float, default=2e-1, help="Relative tolerance for correctness checks")
    parser.add_argument("--atol", type=float, default=2e-1, help="Absolute tolerance for correctness checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device required for TileLang tests")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    configs = [
        TestConfig(batch_size=24, tokens_per_batch=1, f_in=1024, rank=32, f_out=4096, num_loras=5, num_layers=3, layer_idx=1, scale=0.6, accumulate=True),
        TestConfig(batch_size=16, tokens_per_batch=4, f_in=8192, rank=8, f_out=16384, num_loras=6, num_layers=2, layer_idx=0, scale=1.0, accumulate=True),
        TestConfig(batch_size=64, tokens_per_batch=512, f_in=1024, rank=16, f_out=8192, num_loras=3, num_layers=4, layer_idx=2, scale=0.25, accumulate=False),
    ]

    for idx, cfg in enumerate(configs, 1):
        print(f"[Test {idx}] Running {cfg}")
        _run_single_test(cfg, device=device, dtype=dtype, rtol=args.rtol, atol=args.atol)

    print("TileLang SGMV shrink/expand tests passed.")


if __name__ == "__main__":
    main()
