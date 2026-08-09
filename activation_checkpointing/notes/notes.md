# Activation Checkpointing Notes

Notes from walking through `demo.py`, `decoder.py`, and `simple_checkpoint.py`.

---

## 1. Demo glue: unused args and the call counter

### `del name` in `run_strategy`

```python
def run_strategy(name: str, model, original_input, run) -> RunResult:
    del name
    ...
```

`name` is unused inside the function; the caller keys results by name outside. `del name` is an idiom to mark the argument as intentionally ignored (silences unused-arg linters). Same pattern in the hook: `del module, args`.

Cleaner alternatives: drop the parameter, or name it `_name`.

### Forward pre-hook call counter

```python
def count_call(module, args):
    nonlocal calls
    del module, args
    calls += 1

handle = model.register_forward_pre_hook(count_call)
```

PyTorch invokes this right before each `DecoderBlock.forward`. Checkpointing re-runs forward during backward, so:

| Strategy | Typical `calls` |
|---|---|
| Eager | 1 |
| Checkpointed | 2 (original + replay) |

That is the `calls` column in the demo table.

---

## 2. Reentrant vs non-reentrant

`torch.utils.checkpoint.checkpoint(..., use_reentrant=...)` has two implementations.

### Reentrant (`use_reentrant=True`, legacy)

Custom `torch.autograd.Function`:

- **Forward:** run `fn` under `torch.no_grad()`; save inputs.
- **Backward:** re-run `fn` with grad, then call `torch.autograd.backward()` inside backward — re-enters the engine.

Consequences:

- Grads mainly through tensor inputs; closed-over params can silently miss grads (see below).
- No `torch.autograd.grad()`; whole segment always recomputed; awkward with DDP / hooks / nested checkpoint / double backward.

### Non-reentrant (`use_reentrant=False`, modern)

Saved-tensor hooks (`torch.autograd.graph.saved_tensors_hooks`):

- **Forward:** run `fn` with grad; pack hook replaces saved activations with placeholders.
- **Backward:** first unpack triggers replay; then ordinary backward (no nested engine call).

Consequences:

- Grads to everything (including closed-over params).
- Works with `autograd.grad`, DDP, nested checkpoint, double backward.
- Early stopping + selective activation checkpointing (`context_fn`).

`simple_checkpoint.checkpoint` is a teaching reimplementation of the **non-reentrant** path.

### Closed-over params

**Closed-over params** = weight/bias tensors (or any `requires_grad` tensors) that the checkpointed callable uses via Python scoping, not as explicit `checkpoint(fn, *args)` inputs.

Python closes over names from the enclosing scope. Typical forms:

```python
# w is closed over — used inside fn, but not an arg to checkpoint()
w = nn.Parameter(torch.randn(4, 4))

def fn(x):
    return x @ w

y = checkpoint(fn, x)   # only x is a checkpoint input
```

Same idea with a module:

```python
y = checkpoint(lambda x: block(x), x)
# block.weight / bias are closed over via `block` / `self`, not passed as args
```

vs the explicit form:

```python
y = checkpoint(fn, x, w)  # w is a real input; reentrant path can track it
```

Why this matters for **reentrant** checkpoint:

1. Forward runs under `no_grad` and only `save_for_backward`s the tensor `*args`.
2. Backward detaches those args, replays `fn`, then `autograd.backward` and returns grads **only for those args** (`CheckpointFunction.backward`).

Autograd’s contract for the checkpoint node is “grad w.r.t. tensor inputs.” Anything `fn` reaches only by closure is outside that contract. Leaf `nn.Parameter`s often still get `.grad` as a side effect of the nested `backward()`, which is why it can look fine — until it doesn’t:

- no input with `requires_grad`
- closed-over non-leaves / outer activations
- `autograd.grad` (doesn’t populate `.grad`)
- hooks / DDP / nested checkpoint interactions

That’s the “silently miss grads” failure mode. Non-reentrant keeps a real autograd graph through the segment, so closed-over params participate normally.

### Default / why

Historically default was reentrant (it came first). Since PyTorch ~2.1 you must pick explicitly (warning if omitted); recommendation and modern default direction is **`use_reentrant=False`**: more correct and more capable.

---

## 3. `stats=checkpoint_stats` — do you need it?

```python
stats = CheckpointStats() if stats is None else stats
```

If you omit `stats`, `checkpoint()` still works: it allocates a throwaway `CheckpointStats` so hooks always have somewhere to write.

Passing `stats=checkpoint_stats` is an **out-parameter**: the same object accumulates counters so the demo can print `recomputations`, `stopped_early`, intercepted tensor counts after the run.

The selective run intentionally does not pass checkpoint `stats` (avoids polluting standalone numbers); it instruments via `SelectiveCheckpointStats` on the SAC contexts instead.

---

## 4. `_checkpoint_generator` is not √n

This is **one** non-reentrant checkpoint segment, not Chen’s multi-segment √n schedule.

| | Classic Chen √n | This generator |
|---|---|---|
| Scope | Whole net of n layers | One callable `function(*args)` |
| Decision | Which layers are boundaries | Always checkpoint that whole `function` |
| Memory | O(√n) if ~√n segments | Activations inside segment ≈ 0; keep inputs + stubs |
| Placement | Wrap every √n layers | Caller wraps one block (here: one `DecoderBlock`) |

√n is a **policy on top**: wrap segments with `checkpoint(...)`. This file only implements the mechanism for one wrap.

### Generator sandwich

```python
generator = _checkpoint_generator(...)
next(generator)                     # setup → first yield
result = function(*args, **kwargs)  # real forward under hooks
next(generator)                     # mark complete / cleanup
```

Phases inside the generator:

1. Capture rematerialization context: `context_fn()`, RNG, amp.
2. `_NoopSaveInputs.apply(...)` — save **inputs** on the outer graph.
3. `with _CheckpointHooks(frame), forward_context: yield` — forward under pack hooks.
4. Backward (outside generator): first unpack → `recompute_fn` → refill holders.

No “every x stages” loop inside the generator. Rematerialization is **whole segment, once per backward** (with optional early stop).

Conceptual √n usage (not what `demo.py` does):

```python
segment = int(n**0.5)
for i, block in enumerate(blocks):
    if i % segment == 0:
        x = checkpoint(block, x)
    else:
        x = block(x)
```

---

## 5. Goal: do *not* store (most) activations

“Activation checkpointing” is a misnomer. Normal autograd **stores** activations via `save_for_backward`. Checkpointing **refuses** to keep those intermediates and only keeps a cheap checkpoint (segment inputs + RNG/amp), then **recomputes** during backward.

$$
\text{memory} \downarrow,\quad \text{compute} \uparrow \approx +1\times\text{forward for that segment}
$$

| Stored | Why |
|---|---|
| Segment inputs (`_NoopSaveInputs`) | Starting point for rematerialization |
| RNG / amp state | Replay must match (dropout, etc.) |
| `_Holder` stubs | Graph topology / “need a tensor here later” |
| (optional) SAC-selected ops | Keep expensive/hot ones; recompute the rest |

---

## 6. Where storage and recomputation live in code

### What is kept

**Inputs** — durable tensor store for the segment:

- `_NoopSaveInputs.setup_context` → `ctx.save_for_backward(...)`
- Wired at: `frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, *args)`

**RNG / amp** — top of `_checkpoint_generator` (`cpu_rng_state`, `device_rng_states`, autocast kwargs).

**Placeholders instead of activations** — `_CheckpointHooks.pack_hook`:

```python
def pack_hook(tensor: torch.Tensor) -> _Holder:
    holder = _Holder()
    frame.weak_holders.append(weakref.ref(holder))
    ...
    return holder  # autograd saves this, not `tensor`
```

**Selective AC** — `_CachingTorchDispatchMode` appends to `storage[func]` when policy is SAVE (e.g. cache `bmm`).

### Where unstored stuff is recomputed

`_CheckpointHooks.unpack_hook` on first unpack:

1. `inputs = ctx.get_args(ctx.saved_tensors)`
2. `frame.stats.recomputations += 1`
3. `frame.recompute_fn(*inputs)` under `_RecomputationHooks` + `enable_grad`
4. `_RecomputationHooks.pack_hook` fills `frame.recomputed[handle] = tensor`
5. Optional early stop via `_StopRecomputationError`
6. Unpack returns tensor from `frame.recomputed`

`recompute_fn` restores RNG/amp and calls `function(*saved_args, **saved_kwargs)` again.

Under SAC, SAVE ops pop from `storage` in `_CachedTorchDispatchMode` instead of recomputing; everything else runs `func(*args, **kwargs)` again.

| Thing | Where |
|---|---|
| Store inputs | `_NoopSaveInputs.setup_context` |
| Store RNG/amp | top of `_checkpoint_generator` |
| Drop activations | `_CheckpointHooks.pack_hook` → `_Holder` |
| Optionally keep some ops | `_CachingTorchDispatchMode` → `storage[func]` |
| Recompute dropped stuff | `unpack_hook` → `recompute_fn` → `function(...)` |
| Temporary refill after recompute | `_RecomputationHooks.pack_hook` → `frame.recomputed` |

---

## 7. Mind map: one `DecoderBlock` through `checkpoint`

Treat the block as $f$:

```text
x  ──►  DecoderBlock.forward  ──►  y
         (attn + mlp + dropout)
```

### Phase A — Forward (under pack hooks)

```text
1. Snapshot for rematerialization
   ┌─────────────────────────────────────────┐
   │ KEEP:  x  (via _NoopSaveInputs)         │
   │ KEEP:  RNG (dropout) + amp dtype        │
   │ KEEP:  empty graph stubs (_Holder list) │
   └─────────────────────────────────────────┘

2. Run block with grad ON; pack_hook intercepts every
   tensor autograd would save_for_backward:

   x
   → attn_norm, qkv, rope
   → scores = Q@Kᵀ, softmax, dropout
   → attended = P@V, out_proj
   → residual + dropout
   → mlp_norm, gate/up/silu, down
   → residual + dropout
   → y

   For each "I need this later for backward":
     normal:   save tensor  (MiB-scale activation)
     here:     save _Holder (bytes)  ← activation dropped
```

Examples inside this decoder:

| Op region | Typical saved activations | Standalone checkpoint |
|---|---|---|
| `Linear` / matmuls | inputs to GEMMs | dropped → `_Holder` |
| `softmax` | pre-softmax / probs | dropped |
| `dropout` | mask (or seed via RNG) | dropped; RNG kept at boundary |
| `silu`, residuals | inputs | dropped |
| Attention `@` / `bmm` | Q,K / P,V style saves | dropped unless SAC caches them |

After forward:

```text
GPU memory that matters
├── parameters (always)
├── x          ← the checkpoint
├── y          ← output (needed by next layer / loss)
├── autograd graph with _Holders instead of activations
└── RNG/amp metadata on the frame
```

### Phase B — Backward (lazy rematerialize, then normal Grad)

```text
                    first unpack_hook fires
                              │
                              ▼
              ┌───────────────────────────────┐
              │ REPLAY (exactly once)         │
              │  restore RNG + amp            │
              │  recompute_fn:                │
              │      y' = DecoderBlock(x)     │  ← same forward code path
              │  each save point again:       │
              │      pack → frame.recomputed  │  ← refill temporary cache
              │  early_stop when all holders  │
              │      filled (may cut mid-MLP) │
              └───────────────────────────────┘
                              │
                              ▼
              unpack returns real tensors from frame.recomputed
                              │
                              ▼
              ordinary .backward() through attn/mlp
              → grads into params + x.grad
```

Recomputation is not “layer 7 of 32.” It is: **re-run this whole `DecoderBlock` from saved `x`**, with the same dropout RNG, long enough to recreate the saved tensors this backward needs.

### Strategy comparison for this demo

```text
EAGER
  forward:  compute + STORE [norm, QKV, scores, P, V, mlp intermediates, ...]
  backward: use stored tensors
  cost: memory high, compute 1×fwd

CHECKPOINT (standalone)
  forward:  compute + STORE only [x] + holders; DROP intermediates
  backward: REPLAY block(x) → recreate intermediates → then Grad
  cost: memory ~ inputs+output+graph, compute ~2×fwd for the block

SELECTIVE (cache bmm)
  forward:  DROP most; KEEP bmm/matmul outputs in SAC storage
  backward: REPLAY block(x), but bmm results POP from cache
  cost: less recompute FLOPs, some extra memory for cached bmms
```

In this attention impl, heavy `@` ops include `query @ key.T` and `probabilities @ value`. SAC keeps those; still recomputes softmax, norms, etc. around them as needed.

### One sentence

**Stored:** segment input `x` (+ RNG/amp) and tiny holders.  
**Not stored:** the block’s internal activations.  
**Recomputed:** by calling `DecoderBlock.forward(x)` again inside `unpack_hook` → `recompute_fn`, then backward uses the rematerialized tensors as if they had been saved all along.

---

## 8. Selective AC: `create_selective_checkpoint_contexts`

General checkpointing drops **all** intermediates inside the segment. SAC sits **on top of that same mechanism** and says: for a chosen subset of ops, keep the outputs anyway and reuse them on replay.

### What the factory returns

```python
def create_selective_checkpoint_contexts(...):
    storage: defaultdict[Any, list[Any]] = defaultdict(list)
    stats = SelectiveCheckpointStats() if stats is None else stats
    return (
        _CachingTorchDispatchMode(policy_fn, storage, stats),   # forward
        _CachedTorchDispatchMode(...),                          # recompute
    )
```

It builds a **paired** `(forward_context, recompute_context)` that share one FIFO `storage` keyed by `OpOverload`. That pair is exactly what `context_fn` returns for non-reentrant checkpoint:

```python
forward_context, recompute_context = context_fn()
# forward:  with _CheckpointHooks, forward_context
# replay:   with _RecomputationHooks, recompute_context
```

Demo wiring:

```python
sac_contexts = create_selective_checkpoint_contexts(
    [torch.ops.aten.bmm.default], stats=sac_stats
)
checkpoint(model, value, context_fn=lambda: sac_contexts)
```

Policy from an op list: listed ops → `MUST_SAVE`, everything else → `PREFER_RECOMPUTE`.

### Layering: SAC vs plain checkpoint

```text
                    plain checkpoint                 SAC
────────────────────────────────────────────────────────────────
Outer mechanism     same                             same
                    save inputs x                    save inputs x
                    pack → _Holder (drop acts)       pack → _Holder (drop acts)
                    unpack → recompute_fn            unpack → recompute_fn

Inside the segment  nullcontext / nullcontext        CachingMode / CachedMode
                    every op runs                    SAVE ops: cache / reuse
                    every op recomputed on replay    RECOMPUTE ops: run again
```

SAC does **not** replace pack/unpack holders. Autograd still doesn’t keep the usual `save_for_backward` activations. SAC adds a **second, op-level cache** via `TorchDispatchMode` around the region.

### Forward: `_CachingTorchDispatchMode`

For each Aten op (skipping metadata-ish ops in `_SAC_IGNORED_OPS`):

1. Ask `policy_fn(ctx, op, *args, **kwargs)`.
2. Always run `output = func(*args, **kwargs)`.
3. If policy is `MUST_SAVE` / `PREFER_SAVE`: detach (aliasing-aware), wrap in `_VersionWrapper`, `storage[func].append(...)`.
4. If `PREFER_RECOMPUTE` / `MUST_RECOMPUTE`: don’t cache.

For `[aten.bmm.default]`: every `bmm` result is pushed into a FIFO; softmax, linear internals, etc. are not. Cached `bmm` outputs live in `storage`, not in the `_Holder`s.

### Recompute: `_CachedTorchDispatchMode`

On replay of `DecoderBlock(x)`:

1. Same policy.
2. If SAVE: **don’t** run the op — `storage[func].pop(0)` and return that tensor (order must match forward).
3. If RECOMPUTE: run `func(*args, **kwargs)` for real.

Replay still walks the block, but selected ops are skipped; cheaper glue is rematerialized. Demo stats expose both `cached bmm calls` and `bmm results reused during replay`.

### Mental picture for this decoder

```text
plain standalone
  keep:     x
  drop:     everything inside (scores, P, V products, mlp acts, ...)
  replay:   full DecoderBlock(x) again

SAC (cache bmm)
  keep:     x
  also keep: outputs of aten.bmm.default (FIFO in storage)
  drop:     everything else (still holders for autograd saves)
  replay:   DecoderBlock(x) again, but each bmm = pop cache
            softmax / norms / most of MLP path still execute
```

### Tradeoff vs plain AC

| | Plain | SAC |
|---|---|---|
| Memory after forward | lowest (inputs + holders) | higher by `cached_tensor_bytes` |
| Recompute FLOPs | ~full segment forward | segment forward minus cached ops |
| Correctness constraint | same graph order | same + policy/order stable; mutation of cached tensors forbidden unless allowed |

### Why a factory that returns two modes

- Forward needs **write** semantics (`append`).
- Replay needs **read** semantics (`pop`) and must not re-append into the same FIFO.

Sharing `storage` couples them. `context_fn` is the hook non-reentrant checkpoint already had for “run this around forward / around recompute”; SAC is the interesting `context_fn`.

**Bottom line:** general checkpointing = don’t let autograd retain activations; rematerialize from inputs. SAC = same, but **selectively pin** some op outputs so rematerialization can skip those ops.

---

## 9. Measuring backward FLOPs (SAC memory↑, compute↓)

`demo.py` wraps `.backward()` in `torch.utils.flop_counter.FlopCounterMode` and reports `bwd GFLOPs`. That counter covers **rematerialization + true backward** (not the original forward).

Typical cuda / bf16 run at default shape `(2, 256, 512)`:

| Strategy | fwd MiB | bwd GFLOPs |
|---|---:|---:|
| eager | 51.1 | 7.516 |
| standalone | 0.5 | 11.274 |
| selective (cache bmm) | 3.0 | 11.006 |

SAC pays **+2.5 MiB** forward memory (the cached `bmm` outputs) and saves FLOPs on rematerialization because those `bmm`s are popped from cache instead of re-executed. Eager is lowest compute (no rematerialize); checkpointed runs pay ~extra forward FLOPs on replay.

### Math: delta is exactly the two attention `bmm`s

Demo defaults: $B=2$, $S=256$, $H=8$, $d=D/H=64$.

Attention matmuls in `CausalSelfAttention`:

1. $Q K^{\top}$: $(B,H,S,d)\,@\,(B,H,d,S)$ → `aten.bmm`
2. $P V$: $(B,H,S,S)\,@\,(B,H,S,d)$ → `aten.bmm`

`FlopCounterMode` uses $2MKN$ for $(M,K)\,@\,(K,N)$ (mul+add):

$$
\begin{aligned}
\mathrm{FLOPs}(QK^{\top})
&= (BH)\cdot 2\cdot S\cdot d\cdot S \\
&= 16\cdot 2\cdot 256\cdot 64\cdot 256 \\
&= 134{,}217{,}728 \\
\mathrm{FLOPs}(PV)
&= (BH)\cdot 2\cdot S\cdot S\cdot d \\
&= 16\cdot 2\cdot 256\cdot 256\cdot 64 \\
&= 134{,}217{,}728 \\
\mathrm{FLOPs}(QK^{\top})+\mathrm{FLOPs}(PV)
&= 268{,}435{,}456 \\
&= 0.268435456\ \mathrm{GFLOPs}
\end{aligned}
$$

Measured raw totals (not the 3-decimal table rounding):

| | FLOPs | GFLOPs |
|---|---:|---:|
| standalone bwd phase | 11,274,289,152 | 11.274289152 |
| selective bwd phase | 11,005,853,696 | 11.005853696 |
| **delta** | **268,435,456** | **0.268435456** |

$$
\begin{aligned}
\Delta
&= 11{,}274{,}289{,}152 - 11{,}005{,}853{,}696 \\
&= 268{,}435{,}456 \\
&= 2\cdot(BH)\cdot(2SdS)
\end{aligned}
$$

Integer identity: the SAC vs standalone backward-phase FLOP gap equals the analytical cost of the two forward attention `bmm`s. The printed `11.274 − 11.006 = 0.268` is just rounding of those exact GFLOP values.

---

## Sources

- `simple_checkpoint.py`, `demo.py`, `decoder.py` in this directory
- PyTorch docs: [`torch.utils.checkpoint`](https://pytorch.org/docs/stable/checkpoint.html)
- PyTorch non-reentrant design: [pytorch#69508](https://github.com/pytorch/pytorch/pull/69508)
- Chen et al., “Training Deep Nets with Sublinear Memory Cost” ([arXiv:1604.06174](https://arxiv.org/abs/1604.06174))
