"""Run eager, native, standalone, and selective checkpointing side by side."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from decoder import DecoderBlock, DecoderConfig
from simple_checkpoint import (
    CheckpointStats,
    SelectiveCheckpointStats,
    checkpoint,
    create_selective_checkpoint_contexts,
)


@dataclass
class RunResult:
    output: torch.Tensor
    input_gradient: torch.Tensor
    module_calls: int
    forward_memory_mib: float
    peak_memory_mib: float


def run_strategy(
    name: str,
    model: DecoderBlock,
    original_input: torch.Tensor,
    run: Callable[[torch.Tensor], torch.Tensor],
) -> RunResult:
    del name
    model.zero_grad(set_to_none=True)
    x = original_input.detach().clone().requires_grad_(True)
    torch.manual_seed(1234)
    if x.is_cuda:
        torch.cuda.manual_seed_all(1234)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(x.device)
        baseline = torch.cuda.memory_allocated(x.device)
    else:
        baseline = 0

    calls = 0

    def count_call(module, args):
        nonlocal calls
        del module, args
        calls += 1

    handle = model.register_forward_pre_hook(count_call)
    output = run(x)
    forward_memory = (
        torch.cuda.memory_allocated(x.device) - baseline if x.is_cuda else 0
    )
    output.float().square().mean().backward()
    peak_memory = (
        torch.cuda.max_memory_allocated(x.device) - baseline if x.is_cuda else 0
    )
    handle.remove()
    assert x.grad is not None
    return RunResult(
        output.detach().float().cpu(),
        x.grad.detach().float().cpu(),
        calls,
        forward_memory / 2**20,
        peak_memory / 2**20,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    args = parser.parse_args()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    config = DecoderConfig(
        dim=args.dim,
        num_heads=args.heads,
        hidden_dim=args.hidden_dim,
        dropout=0.1,
    )
    torch.manual_seed(0)
    model = DecoderBlock(config).to(device=device, dtype=dtype).train()
    x = torch.randn(args.batch, args.sequence, args.dim, device=device, dtype=dtype)

    checkpoint_stats = CheckpointStats()
    sac_stats = SelectiveCheckpointStats()
    sac_contexts = create_selective_checkpoint_contexts(
        [torch.ops.aten.bmm.default], stats=sac_stats
    )

    runs = {
        "eager": lambda value: model(value),
        "torch non-reentrant": lambda value: torch_checkpoint(
            model, value, use_reentrant=False
        ),
        "standalone": lambda value: checkpoint(
            model, value, stats=checkpoint_stats
        ),
        "selective (cache bmm)": lambda value: checkpoint(
            model, value, context_fn=lambda: sac_contexts
        ),
    }

    results: dict[str, RunResult] = {}
    for name, strategy in runs.items():
        results[name] = run_strategy(name, model, x, strategy)

    eager = results["eager"]
    print(f"device={device}, dtype={dtype}, shape={tuple(x.shape)}")
    print(
        f"{'strategy':<24} {'calls':>5} {'fwd MiB':>10} {'peak MiB':>10} "
        f"{'max |output diff|':>18} {'max |grad diff|':>16}"
    )
    for name, result in results.items():
        output_error = (result.output - eager.output).abs().max().item()
        gradient_error = (result.input_gradient - eager.input_gradient).abs().max().item()
        print(
            f"{name:<24} {result.module_calls:>5} "
            f"{result.forward_memory_mib:>10.1f} {result.peak_memory_mib:>10.1f} "
            f"{output_error:>18.3e} {gradient_error:>16.3e}"
        )

    print("\nstandalone checkpoint internals")
    print(f"  replay count: {checkpoint_stats.recomputations}")
    print(f"  early stopped: {checkpoint_stats.stopped_early}")
    print(
        "  forward tensors replaced by holders: "
        f"{checkpoint_stats.forward_saved_tensors_intercepted} "
        f"({checkpoint_stats.forward_saved_tensor_bytes_intercepted / 2**20:.1f} MiB logical)"
    )
    print("\nselective checkpoint internals")
    print(f"  cached bmm calls: {sac_stats.cached_ops['aten.bmm.default']}")
    print(f"  bmm results reused during replay: {sac_stats.reused_ops['aten.bmm.default']}")
    print(f"  selective cache size: {sac_stats.cached_tensor_bytes / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
