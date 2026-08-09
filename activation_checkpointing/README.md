# Activation checkpointing, with the machinery exposed

This is a standalone teaching implementation of PyTorch's modern, non-reentrant
activation checkpoint. It uses core autograd and dispatcher primitives, but the
custom path never calls `torch.utils.checkpoint.checkpoint`.

It is intentionally more realistic than the common custom `autograd.Function`
example. That simple reentrant formulation runs forward under `no_grad()` and
reruns the entire function in backward. Current PyTorch recommends the
non-reentrant implementation, whose interesting mechanism is
`saved_tensors_hooks`.

## The central trick

During a normal forward pass, backward formulas call `save_for_backward` for
the tensors they will need later. The custom checkpoint installs hooks that
replace every such tensor with a tiny `_Holder` object:

```text
original forward                        backward's first tensor request
----------------                        -------------------------------
autograd wants to save tensor A         unpack(_Holder for A)
              |                                      |
              v                                      v
pack(A) -> _Holder  -- kept in graph --> replay the checkpointed function
              |                                      |
              `-- A can be freed          replay pack(A') -> handle -> A'
                                                     |
                                                     v
                                           return A' to the backward formula
```

`_NoopSaveInputs` anchors only the checkpoint boundary inputs to the outer
autograd graph. On the first unpack, the function is replayed with grad mode
enabled. A second set of saved-tensor hooks catches the replayed activations and
matches them, in order, to the holders from the first pass.

The implementation keeps the less-obvious production behavior that matters for
correctness:

- CPU and accelerator RNG states are forked and restored, so dropout produces
  the same mask during replay without advancing the caller's RNG twice.
- CPU and accelerator autocast settings are replayed.
- Replay verifies the number and shape/dtype/device metadata of saved tensors.
- An internal exception stops replay immediately after the final needed save
  point, instead of necessarily finishing the function.
- Inputs are retained by an autograd node rather than a plain Python closure.

## Selective activation checkpointing

Selective activation checkpointing (SAC) is a layer around the same replay.
Two paired `TorchDispatchMode`s intercept individual ATen operations:

1. The forward mode asks a policy whether an operation should be saved or
   recomputed. Saved outputs go into a per-`OpOverload` FIFO cache.
2. The replay mode asks the same policy with `is_recompute=True`. It either runs
   the operation or pops the corresponding forward result from the cache.
3. Cached tensors retain their version counter. If later code mutates one
   in-place, replay fails instead of silently using a corrupted result.

`MUST_*` versus `PREFER_*` matters to compiler partitioners in full PyTorch. In
this eager-only implementation they have the same runtime behavior, but both
forms remain in the API so the policy reads like the real one.

The demo caches the two batched matrix multiplies in attention while replaying
the cheaper operations:

```python
sac_contexts = create_selective_checkpoint_contexts(
    [torch.ops.aten.bmm.default]
)
output = checkpoint(block, x, context_fn=lambda: sac_contexts)
```

## Realistic decoder workload

`decoder.py` contains a pre-norm decoder block with RMSNorm, fused QKV
projection, rotary embeddings, explicit causal multi-head attention, residual
dropout, and a SwiGLU MLP. Attention is written explicitly rather than through a
fused SDPA kernel so SAC decisions and operation counts remain visible.

Run all four paths side by side:

```bash
uv sync
uv run python demo.py --device cuda
```

The table reports decoder invocations, CUDA allocated memory, and numerical
differences from eager. The detailed section reports how many autograd tensors
were replaced by holders and how many `bmm` results SAC reused.

Run the correctness suite:

```bash
uv run pytest
```

The tests compare outputs, input gradients, and every parameter gradient with
both eager execution and PyTorch's native non-reentrant checkpoint. They also
cover dropout RNG state, early stop, SAC cache reuse, cache mutation detection,
metadata mismatch errors, no-grad behavior, and CUDA bfloat16.

## Map to PyTorch source

| Here | PyTorch 2.11 source concept |
|---|---|
| `_checkpoint_generator` | `_checkpoint_without_reentrant_generator` |
| `_CheckpointFrame` | `_CheckpointFrame` |
| `_CheckpointHooks` | `_checkpoint_hook` |
| `_RecomputationHooks` | `_recomputation_hook` |
| `_NoopSaveInputs` | `_NoopSaveInputs` |
| `_CachingTorchDispatchMode` | same name |
| `_CachedTorchDispatchMode` | same name |
| `_VersionWrapper` | same name |

The research trail is the current [DeepWiki PyTorch index](https://deepwiki.com/pytorch/pytorch),
the pinned [PyTorch 2.11 checkpoint source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/checkpoint.py),
the [checkpoint API documentation](https://docs.pytorch.org/docs/stable/checkpoint.html),
and PyTorch's [activation-checkpointing techniques overview](https://pytorch.org/blog/activation-checkpointing-techniques/).

## Deliberate boundaries

The code is faithful to the eager mechanism, not a drop-in production clone.
PyTorch additionally handles overlapping and repeated backward graph tasks,
nested checkpoints, `torch.compile` annotations, debug operator traces,
arbitrary accelerator backends, global device contexts, and more elaborate
error reporting. PyTorch 2.11 also exposes CPU-offload policy values; this
example focuses on the save-versus-recompute SAC mechanism. The standalone
implementation therefore supports eager execution and one backward traversal
per checkpointed graph.

The implementation was independently reduced from PyTorch's BSD-licensed
checkpoint module; see `THIRD_PARTY_NOTICE.md`.
