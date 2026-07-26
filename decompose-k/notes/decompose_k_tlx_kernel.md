# TLX Decompose-K Kernel: Detailed Walkthrough

This note explains how
[`decompose_k_tlx_kernel.py`](../kernels/decompose_k_tlx_kernel.py) works, what
the TLX-specific operations mean, how the asynchronous pipeline is synchronized,
and how this implementation improves on
[`decompose_k_triton_kernel_optimized.py`](../kernels/decompose_k_triton_kernel_optimized.py).

The implementation is specialized for small-`M`/`N`, large-`K` matrix
multiplications on Hopper GPUs using FP16 or BF16 inputs and FP32 accumulation.

## Contents

- [High-level idea](#high-level-idea)
- [How the launch decomposes the matmul](#how-the-launch-decomposes-the-matmul)
- [Program and tile mapping](#program-and-tile-mapping)
- [Shared-memory stage allocation](#shared-memory-stage-allocation)
- [`tlx.local_view`](#tlxlocal_view)
- [Asynchronous load pipeline](#asynchronous-load-pipeline)
- [`async_load_wait_group`](#async_load_wait_group)
- [Asynchronous WGMMA pipeline](#asynchronous-wgmma-pipeline)
- [`async_dot_wait`](#async_dot_wait)
- [Complete two-stage pipeline](#complete-two-stage-pipeline)
- [Single-iteration fast path](#single-iteration-fast-path)
- [Single-launch split reduction](#single-launch-split-reduction)
- [Why the atomic add happens before the load](#why-the-atomic-add-happens-before-the-load)
- [Workspace initialization and reuse](#workspace-initialization-and-reuse)
- [Host-side configuration](#host-side-configuration)
- [How TLX improves over optimized Triton](#how-tlx-improves-over-optimized-triton)
- [Optimized Triton reducer context](#optimized-triton-reducer-context)
- [Tradeoffs and correctness assumptions](#tradeoffs-and-correctness-assumptions)
- [Worked example](#worked-example)
- [Compact end-to-end pseudocode](#compact-end-to-end-pseudocode)

## High-level idea

For a normal matrix multiplication,

```text
C = A @ B

A: [M, K]
B: [K, N]
C: [M, N]
```

Decompose-K divides the `K` dimension into `SPLIT_K` independent ranges:

```text
K = K_0 + K_1 + ... + K_(SPLIT_K - 1)
```

Each split computes a partial output:

```text
partial_s = A[:, K_s] @ B[K_s, :]
```

The final output is:

```text
C = partial_0 + partial_1 + ... + partial_(SPLIT_K - 1)
```

The TLX kernel does all of the following in one kernel launch:

1. Assign one program to each `(output tile, K split)` pair.
2. Stage A and B tiles through shared memory with asynchronous loads.
3. Compute the partial tile using asynchronous Hopper WGMMA operations.
4. Atomically add the partial tile into one FP32 accumulator workspace.
5. Use a counter to identify the last arriving split for each output tile.
6. Let that last program load the complete tile, apply the optional ReLU, and
   write `C`.
7. Clear the accumulator and counter so the workspace can be reused.

The launch structure is therefore:

```text
grid = (number_of_output_tiles, SPLIT_K)
```

See the
[`_decompose_k_tlx_atomic` kernel](../kernels/decompose_k_tlx_kernel.py#L36-L184)
and its
[`grid` construction](../kernels/decompose_k_tlx_kernel.py#L279-L309).

## How the launch decomposes the matmul

The two program IDs have different responsibilities:

```python
pid = tl.program_id(0)
split_id = tl.program_id(1)
```

- `pid` selects an `M x N` output tile.
- `split_id` selects one contiguous portion of `K`.

The amount of K owned by one split is:

```python
k_per_split = K // SPLIT_K
split_start = split_id * k_per_split
```

So split `s` owns:

```text
[s * k_per_split, (s + 1) * k_per_split)
```

For example, with:

```text
K = 8192
SPLIT_K = 8
```

each split owns `1024` K elements:

```text
split 0: [   0, 1024)
split 1: [1024, 2048)
...
split 7: [7168, 8192)
```

The host wrapper requires `K % SPLIT_K == 0`, so K is divided evenly between
splits. A split can still have a partial final `BLOCK_K` iteration, which is
handled by the K mask.

See the
[`k_per_split` and pointer construction](../kernels/decompose_k_tlx_kernel.py#L70-L85).

## Program and tile mapping

The linear output-tile `pid` is mapped to `(pid_m, pid_n)` using grouped-M
ordering:

```python
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n

group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

This changes scheduling order, not the mathematical result. Programs within a
group advance through nearby M tiles before moving farther away, improving the
chance of reusing A-side data through the cache hierarchy.

The program computes:

```text
rows = pid_m * BLOCK_M + [0, ..., BLOCK_M - 1]
cols = pid_n * BLOCK_N + [0, ..., BLOCK_N - 1]
```

M, N, and K masks handle boundary tiles.

## Shared-memory stage allocation

The kernel explicitly allocates staged local buffers:

```python
buffers_a = tlx.local_alloc(
    (BLOCK_M, BLOCK_K),
    tlx.dtype_of(a),
    NUM_STAGES,
)

buffers_b = tlx.local_alloc(
    (BLOCK_K, BLOCK_N),
    tlx.dtype_of(b),
    NUM_STAGES,
)
```

Conceptually:

```text
buffers_a: [NUM_STAGES, BLOCK_M, BLOCK_K]
buffers_b: [NUM_STAGES, BLOCK_K, BLOCK_N]
```

These are shared-memory-backed ring buffers. While one stage is consumed by
WGMMA, another stage can be filled from global memory.

The currently searched configurations all use:

```text
NUM_STAGES = 2
```

so this is a double-buffered pipeline.

See the
[`local_alloc` calls](../kernels/decompose_k_tlx_kernel.py#L87-L92).

## `tlx.local_view`

`tlx.local_view(storage, index)` selects one stage from a multi-stage local
allocation.

For example:

```python
a_stage = tlx.local_view(buffers_a, stage)
```

means:

```text
view stage `stage` of buffers_a as one [BLOCK_M, BLOCK_K] A tile
```

It does **not**:

- copy the data;
- allocate another buffer;
- load anything from global memory; or
- synchronize outstanding asynchronous operations.

It creates a view/reference to the selected stage of the existing local
allocation. Different indices select different shared-memory regions:

```text
local_view(buffers_a, 0) -> A shared-memory stage 0
local_view(buffers_a, 1) -> A shared-memory stage 1
```

The same stage can be passed as:

- the destination of `tlx.async_load`; or
- the operand of `tlx.async_dot`.

Correctness depends on the load and dot wait operations ensuring that the stage
is ready before reading it and no longer in use before overwriting it.

## Asynchronous load pipeline

An asynchronous load is issued separately for A and B:

```python
token_a = tlx.async_load(a_ptrs, a_stage, mask=...)
token_b = tlx.async_load(b_ptrs, b_stage, mask=...)
```

The two operations are then committed as one load group:

```python
tlx.async_load_commit_group([token_a, token_b])
```

This grouping is useful because the corresponding A and B tiles must both be
ready before WGMMA consumes the stage.

The tokens represent the issued operations. Committing them packages the
previous loads into an asynchronous copy group whose completion can be managed
with `async_load_wait_group`.

The prologue preloads:

```text
NUM_STAGES - 1
```

stages. With the current two-stage configuration, that means it initially loads
stage 0. The main loop then consumes the ready stage and fills the other stage.

See the
[`async_load` prologue](../kernels/decompose_k_tlx_kernel.py#L94-L114).

## `async_load_wait_group`

The conceptual signature is:

```python
tlx.async_load_wait_group(pendings, tokens=None)
```

It waits on committed asynchronous **copy/load groups**.

### Meaning of `pendings`

`pendings` is the maximum number of newer committed load groups that may still
be outstanding when the function returns.

| Call | Meaning |
| --- | --- |
| `async_load_wait_group(0)` | Drain all previously committed load groups. |
| `async_load_wait_group(1)` | Older groups must finish, but one newer group may remain in flight. |
| `async_load_wait_group(2)` | Older groups must finish, but two newer groups may remain in flight. |

The argument is **not**:

- a group ID;
- a stage index;
- a number of loads to execute;
- a number of groups to wait for; or
- a cycle count.

The counterintuitive but important point is:

```text
wait_group(0) is stricter than wait_group(1)
```

### Meaning of `tokens`

`tokens` optionally supplies explicit asynchronous dependency handles. The TLX
kernel does not pass this argument, so the wait applies to previously committed
load groups tracked by the program.

### Value used by this kernel

The steady-state loop calls:

```python
tlx.async_load_wait_group(NUM_STAGES - 2)
```

For the current `NUM_STAGES = 2` configurations:

```text
NUM_STAGES - 2 = 0
```

Therefore, the actual call is:

```python
tlx.async_load_wait_group(0)
```

Before the kernel passes `a_stage` and `b_stage` to WGMMA, all committed loads
have completed and the selected stage is ready.

With a hypothetical three-stage pipeline, `wait_group(1)` could leave one newer
prefetch group in flight while ensuring the older stage being consumed is
complete.

See the
[`async_load_wait_group` call](../kernels/decompose_k_tlx_kernel.py#L124-L129).

## Asynchronous WGMMA pipeline

Once a shared-memory A/B stage is ready, the kernel issues:

```python
acc = tlx.async_dot(a_stage, b_stage, acc)
```

On Hopper, this follows the warp-group matrix multiply-accumulate path. The
operation conceptually performs:

```text
acc += a_stage @ b_stage
```

using FP32 accumulation.

The operation is asynchronous: issuing `async_dot` does not necessarily mean
that every result has already been committed to `acc` or that its shared-memory
operands can immediately be overwritten.

The current configuration uses four warps, which form one warp group for the
Hopper WGMMA operation.

## `async_dot_wait`

The signature used by the kernel is:

```python
acc = tlx.async_dot_wait(pendings, acc)
```

It waits on outstanding asynchronous **dot/WGMMA groups**.

### Meaning of `pendings`

The integer has the same outstanding-group convention as the load wait:

| Call | Meaning |
| --- | --- |
| `async_dot_wait(0, acc)` | Drain every outstanding asynchronous dot group. |
| `async_dot_wait(1, acc)` | Older dot groups must complete, but the newest group may remain in flight. |

Again:

```text
async_dot_wait(1) does not mean "wait for one dot"
```

It means:

```text
return once no more than one dot group remains outstanding
```

### Meaning of `acc`

The second argument is the accumulator tensor and dependency carrier.

It is not:

- a pending-operation count;
- a stage number;
- a shared-memory buffer; or
- a separate output tensor.

Passing `acc` orders the wait with respect to the asynchronous WGMMA chain that
produces that accumulator. The call returns the accumulator value after adding
the required dependency, so the result is reassigned:

```python
acc = tlx.async_dot_wait(1, acc)
```

### Why the loop uses `async_dot_wait(1)`

After issuing the current WGMMA, the kernel identifies the shared-memory stage
that will be refilled:

```python
next_iter = k_iter + NUM_STAGES - 1
next_index = next_iter % NUM_STAGES
```

It then calls:

```python
acc = tlx.async_dot_wait(1, acc)
```

For a two-stage ring, this ensures older WGMMA work that might still reference
the stage being reused has completed. The newest WGMMA is allowed to continue
while the other stage is refilled, preserving compute/load overlap.

At the end of the K loop:

```python
acc = tlx.async_dot_wait(0, acc)
```

drains all remaining WGMMA work. The accumulator is then final and can safely be
atomically reduced into global memory.

See the
[`async_dot_wait(1)` reuse point](../kernels/decompose_k_tlx_kernel.py#L131-L147)
and the
[`async_dot_wait(0)` final drain](../kernels/decompose_k_tlx_kernel.py#L151-L155).

### Load groups and dot groups are independent

These waits control different asynchronous domains:

| Wait | What it makes safe |
| --- | --- |
| `async_load_wait_group(...)` | Reading A/B values from a shared-memory stage |
| `async_dot_wait(...)` | Using accumulator results and reusing shared-memory stages previously consumed by WGMMA |

A load wait does not drain WGMMA, and a dot wait does not make a pending global
to shared-memory copy complete.

## Complete two-stage pipeline

With `NUM_STAGES = 2`, the steady-state behavior is approximately:

```text
Prologue:
    async-load A0 and B0 into stage 0
    commit copy group 0

Iteration 0:
    load_wait_group(0)
        stage 0 is ready

    async_dot(stage 0)
        WGMMA 0 may remain in flight

    dot_wait(1)
        WGMMA 0 may remain in flight
        any older WGMMA would have to be complete

    async-load A1 and B1 into stage 1
        this may overlap WGMMA 0
    commit copy group 1

Iteration 1:
    load_wait_group(0)
        stage 1 is ready

    async_dot(stage 1)
        WGMMA 1 may remain in flight

    dot_wait(1)
        WGMMA 0 must now be complete
        WGMMA 1 may remain in flight

    refill stage 0
        safe because WGMMA 0 no longer uses it

Continue alternating stage 0 and stage 1.

Epilogue:
    dot_wait(0)
        all WGMMA is complete
        acc is final
```

The core ring-buffer invariant is:

```text
Do not read a stage before its async load completes.
Do not overwrite a stage while an older WGMMA may still consume it.
```

## Single-iteration fast path

The number of K-block iterations for one split is:

```python
num_k_iters = tl.cdiv(k_per_split, BLOCK_K)
```

When `num_k_iters == 1`, the kernel avoids the ring-buffer refill machinery:

```python
a_stage = tlx.local_view(buffers_a, 0)
b_stage = tlx.local_view(buffers_b, 0)
tlx.async_load_wait_group(0)
acc = tlx.async_dot(a_stage, b_stage, acc)
```

The final common:

```python
acc = tlx.async_dot_wait(0, acc)
```

still drains the dot before the global reduction.

See the
[`num_k_iters == 1` branch](../kernels/decompose_k_tlx_kernel.py#L116-L123).

## Single-launch split reduction

The optimized Triton implementation writes one complete `[M, N]` partial matrix
per K split and launches a second reducer. The TLX implementation instead gives
all splits a shared FP32 accumulator matrix plus one counter per output tile.

The workspace layout is:

```text
FP32 accumulator:
    M * N elements

Per-tile counters:
    ceil(M / BLOCK_M) * ceil(N / BLOCK_N) int32-sized elements
```

The counters occupy extra four-byte slots in the FP32 workspace and are viewed
as `int32`:

```python
accumulator = workspace.flatten()[: M * N].view(M, N)
counters = workspace.view(torch.int32).flatten()[
    M * N : M * N + num_tiles
]
```

A `TensorDescriptor` describes the accumulator and its
`[BLOCK_M, BLOCK_N]` block shape:

```python
accumulator_desc = TensorDescriptor.from_tensor(
    accumulator,
    [BLOCK_M, BLOCK_N],
)
```

Each split program performs:

```python
accumulator_desc.atomic_add(
    [pid_m * BLOCK_M, pid_n * BLOCK_N],
    acc,
)
```

Conceptually:

```text
workspace_tile += this split's FP32 partial tile
```

All `SPLIT_K` programs for the same output tile atomically update the same
workspace tile.

See the
[`workspace construction`](../kernels/decompose_k_tlx_kernel.py#L234-L243) and
[`atomic reduction`](../kernels/decompose_k_tlx_kernel.py#L151-L155).

## Why the atomic add happens before the load

After contributing its tile, every split increments the counter for that output
tile:

```python
previous = tl.atomic_add(
    counters + pid,
    1,
    sem="acq_rel",
    scope="gpu",
)
```

`tl.atomic_add` returns the counter's old value. Exactly one program observes:

```python
previous == SPLIT_K - 1
```

That program changed the counter from:

```text
SPLIT_K - 1 -> SPLIT_K
```

It is therefore the last arriving split for this output tile. "Last" means last
to finish and reach the counter, not necessarily the program whose
`split_id == SPLIT_K - 1`.

Only that program executes:

```python
result = accumulator_desc.load(tile_offset)
```

### Example with `SPLIT_K = 4`

Assume the workspace tile begins at zero and split programs finish in the
arbitrary order `2, 0, 3, 1`:

| Arrival | Operation | Old counter | New counter | Result |
| --- | --- | ---: | ---: | --- |
| split 2 | `workspace += partial_2` | 0 | 1 | Exit |
| split 0 | `workspace += partial_0` | 1 | 2 | Exit |
| split 3 | `workspace += partial_3` | 2 | 3 | Exit |
| split 1 | `workspace += partial_1` | 3 | 4 | Load and finalize |

When split 1 loads the tile:

```text
workspace_tile =
    partial_0
  + partial_1
  + partial_2
  + partial_3
```

The atomic add must happen first because:

1. The last program's own partial must be included.
2. The counter may only announce completion after that partial is complete.
3. Loading before all contributions arrive would read an incomplete sum.

The descriptor atomic add does not return the complete reduced tile. It updates
the global accumulator; the elected final program therefore performs a separate
descriptor load after completion has been established.

### Ordering guarantees

There are two different atomic operations:

| Operation | Purpose |
| --- | --- |
| `accumulator_desc.atomic_add(...)` | Atomically contributes the numerical FP32 tile. |
| `tl.atomic_add(counters + pid, 1, ...)` | Publishes completion and elects the final split. |

For the local Triton version used by this project, descriptor reduction lowers
to an asynchronous Hopper TMA reduction followed by a `TMAStoreWaitOp(0)`.
Therefore, the current program's tile contribution is complete before it
increments the counter.

The counter uses:

```text
sem="acq_rel"
scope="gpu"
```

- The release part publishes operations completed before the counter increment.
- The acquire part ensures that the final program observes earlier programs'
  published work before loading the accumulator.
- GPU scope makes the synchronization valid between programs/CTAs running on
  different SMs.

Once the counter reaches `SPLIT_K`, no split for that tile has another
contribution left to issue. The elected program can load a stable final tile.

It then:

1. optionally applies ReLU in FP32;
2. stores the result to the FP16/BF16 output;
3. clears the FP32 accumulator tile; and
4. resets the counter to zero with release semantics.

See the complete
[`last-arrival epilogue`](../kernels/decompose_k_tlx_kernel.py#L157-L184).

## Workspace initialization and reuse

The workspace is zeroed on its first use:

```python
def _initialize_workspace_once(workspace):
    if getattr(workspace, "_tlx_atomic_ready", False):
        return
    workspace.zero_()
    workspace._tlx_atomic_ready = True
```

It does not need to be zeroed by the host before every invocation because the
last arriving program for each tile clears:

```text
accumulator tile -> 0
counter          -> 0
```

The ordering is deliberately:

```text
load final result
store C
clear accumulator tile
release-reset counter
```

Resetting the counter last publishes that the tile workspace is clean.

This protocol assumes one in-flight owner of a workspace. Reusing the same
workspace concurrently from multiple streams would mix independent
invocations' partial sums and counters.

## Host-side configuration

The final TLX candidate space is intentionally narrow:

| Setting | Values |
| --- | --- |
| `BLOCK_M` | `64` |
| `BLOCK_N` | `32` |
| `BLOCK_K` | `128`, `256`, or `512` |
| `GROUP_M` | `8` |
| `num_warps` | `4` |
| TLX `NUM_STAGES` | `2` |

See
[`candidate_configs`](../kernels/decompose_k_tlx_kernel.py#L213-L231).

The launch contains two similarly named but different stage settings:

```python
NUM_STAGES=config.num_stages,
num_stages=1,
```

- `NUM_STAGES` is the TLX kernel constexpr that controls the explicit
  shared-memory ring. It is currently `2`.
- launch-level Triton `num_stages=1` disables an additional compiler-managed
  Triton pipeline because this kernel manually controls staging through TLX.

The host wrapper also enforces:

- A and B are FP16 or BF16;
- A, B, and C use the same dtype;
- all inputs and workspace are on the same device;
- `K` is divisible by `split_k`;
- workspace storage is contiguous FP32;
- C is contiguous; and
- the workspace is large enough.

A and B can be strided because their actual strides are passed into the kernel.

The plugin is configured and loaded through
[`tlx_plugin.py`](../kernels/tlx_plugin.py), and the benchmark configures it
before importing Triton in
[`bench_decompose_k.py`](../bench_decompose_k.py#L11-L19).

## How TLX improves over optimized Triton

The key change is not merely replacing `tl.dot` with a different dot
instruction. TLX changes both the matmul feed pipeline and the split-reduction
algorithm.

| Aspect | Optimized Triton | TLX |
| --- | --- | --- |
| Matmul input path | Regular masked `tl.load` plus `tl.dot`; staging is compiler managed. | Explicit shared-memory ring, asynchronous loads, and Hopper WGMMA. |
| Split output | Writes every partial to `[SPLIT_K, M, N]`. | Atomically adds every partial into one `[M, N]` FP32 accumulator. |
| Reduction | Launches a second flattened-output reducer. | Last arriving split finalizes the tile inside the original launch. |
| Kernel launches | Two | One |
| Main workspace | `SPLIT_K * M * N` FP32 elements | `M * N` FP32 elements plus one counter per tile |
| ReLU | Applied by the second reducer after `tl.sum`. | Applied by the last arriving split after loading the complete FP32 tile. |
| Supported dtype/path | Includes an FP32 path and a non-contiguous reducer fallback. | Hopper WGMMA path is currently FP16/BF16 only and requires contiguous C/workspace. |

### Removed costs

TLX removes:

- the second kernel launch;
- materialization of a separate FP32 matrix for every split;
- reading the complete `[SPLIT_K, M, N]` partial tensor in the reducer; and
- the reducer's explicit `tl.sum` across the split dimension.

### Added costs

TLX adds:

- contending FP32 atomic reductions into the shared accumulator;
- one counter atomic per `(tile, split)`;
- one final counter exchange per tile;
- accumulator cleanup stores; and
- Hopper-specific shared-memory/WGMMA constraints.

The benchmark measures the net result of all these changes together. It does
not isolate how much speedup comes individually from:

- eliminating the second launch;
- reducing workspace traffic;
- explicit asynchronous staging;
- WGMMA;
- different tile configurations; or
- atomic contention.

## Optimized Triton reducer context

The previous optimized Triton implementation launches the partial matmul over:

```text
(number_of_output_tiles, SPLIT_K)
```

and then launches a separate one-dimensional reducer over flattened output
elements:

```text
reduce_grid = (ceil(M * N / XBLOCK),)
```

Inside each reducer program:

```python
x = x_base[:, None]               # [XBLOCK, 1]
r = tl.arange(0, RBLOCK)[None, :] # [1, RBLOCK]

vals = tl.load(partials + r * stride_ps + x, ...)
acc = tl.sum(vals, axis=1)
```

Broadcasting produces a logical pointer tensor of shape:

```text
[XBLOCK, RBLOCK]
```

One source-level `tl.load` therefore represents many scalar loads:

```text
up to XBLOCK * SPLIT_K valid values
```

It is not one scalar load or one global-memory transaction.

`RBLOCK` is the next power of two greater than or equal to `SPLIT_K`. It is the
compile-time reduction width:

```text
RBLOCK = next_power_of_2(SPLIT_K)
```

Padded lanes where:

```text
r >= SPLIT_K
```

are masked and replaced with zero before `tl.sum`.

There is no second launch-grid dimension for `r` because each reducer program
already expresses the complete split dimension as tensor lanes. Launching
different programs for different split ranges would require another
cross-program reduction or atomic accumulation.

### `_reduce_epilogue_vector` versus `_reduce_epilogue_vector_flat`

Both functions perform the same mathematical reduction. The difference is
addressing generality.

The general
[`_reduce_epilogue_vector`](../kernels/decompose_k_triton_kernel_optimized.py#L23-L60):

- converts flat `x` into `(m, n)`;
- reconstructs partial offsets using `stride_pm` and `stride_pn`;
- reconstructs output offsets using `stride_cm` and `stride_cn`; and
- supports non-contiguous partial/output layouts.

The
[`_reduce_epilogue_vector_flat`](../kernels/decompose_k_triton_kernel_optimized.py#L63-L87)
fast path:

- assumes partials and C are contiguous;
- directly addresses `partials + r * stride_ps + x`;
- directly stores to `c + x_base`; and
- avoids division/remainder and extra stride arithmetic.

The host selects the flat reducer only when:

```python
partials.is_contiguous() and c.is_contiguous()
```

TLX removes both reducers from the hot path by completing the split reduction
inside the matmul launch.

## Tradeoffs and correctness assumptions

### Atomic summation order

The order in which split programs reach the accumulator is scheduling
dependent. Floating-point addition is not associative, so the exact FP32 result
can vary slightly between arrival orders.

The explicit optimized Triton reducer has a more structured reduction order.
Both paths can still differ slightly from a reference matmul because their K
decomposition and reduction orders differ.

### Atomic contention

Every split for one output tile updates the same accumulator tile. Large
`SPLIT_K` increases parallelism along K but also increases contention and counter
traffic. More splits are therefore not automatically faster.

### Completion tail

The last arriving program performs extra work:

- descriptor load;
- optional ReLU;
- output store;
- accumulator clear; and
- counter reset.

This creates a per-tile completion tail, although it avoids a second kernel
launch.

### Workspace ownership

The same workspace must not be used by overlapping invocations on different
streams. The one-time initialization and in-kernel reset protocol assumes one
active generation of partials per workspace.

### Supported inputs

The current TLX wrapper:

- accepts FP16/BF16 A and B;
- accumulates in FP32;
- produces FP16/BF16 C;
- requires contiguous C and workspace; and
- does not provide an IEEE FP32 TLX path.

### Masked future prefetches

The steady loop can issue logically future prefetches after the last useful
K block. Their K masks are false, so they do not read invalid elements or affect
the result. This keeps the loop structure simple but can represent some masked
pipeline bookkeeping.

## Worked example

Consider:

```text
M = 64
N = 64
K = 8192

SPLIT_K = 8
BLOCK_M = 64
BLOCK_N = 32
BLOCK_K = 256
NUM_STAGES = 2
```

### Grid

The number of output tiles is:

```text
ceil(64 / 64) * ceil(64 / 32)
= 1 * 2
= 2 tiles
```

The launch grid is:

```text
(2, 8)
```

so the GPU launches `16` programs:

```text
2 output tiles * 8 K splits
```

### Work per split

Each split owns:

```text
k_per_split = 8192 / 8 = 1024
```

Each split performs:

```text
num_k_iters = ceil(1024 / 256) = 4
```

WGMMA iterations using the two-stage shared-memory ring.

For output tile 0 and split 3:

```text
K range = [3 * 1024, 4 * 1024)
        = [3072, 4096)
```

That program computes:

```text
A[0:64, 3072:4096] @ B[3072:4096, 0:32]
```

and atomically adds the resulting `[64, 32]` FP32 tile into output accumulator
tile 0.

### Workspace comparison

Optimized Triton partial storage:

```text
SPLIT_K * M * N
= 8 * 64 * 64
= 32768 FP32 elements
= 131072 bytes
```

TLX workspace:

```text
M * N + number_of_tiles
= 64 * 64 + 2
= 4098 four-byte elements
= 16392 bytes
```

The TLX workspace is about eight times smaller for this example, matching the
expected scaling difference:

```text
optimized Triton: O(SPLIT_K * M * N)
TLX:              O(M * N + number_of_tiles)
```

## Compact end-to-end pseudocode

```text
Host:
    validate A, B, C, workspace, config
    zero workspace on its first use

    accumulator = workspace[0 : M*N] as FP32[M, N]
    counters    = remaining workspace as INT32[num_tiles]

    launch grid = (num_tiles, SPLIT_K)


Each GPU program (tile_id, split_id):
    determine output tile (pid_m, pid_n)
    determine this split's contiguous K range

    allocate NUM_STAGES shared-memory A/B stages
    asynchronously load the prologue stage(s)

    acc = FP32 zeros

    for each BLOCK_K in this split:
        choose current ring-buffer stage
        wait until its async A/B loads are ready
        issue asynchronous WGMMA into acc

        choose the stage that will be refilled
        wait until older WGMMA no longer uses that stage
        asynchronously prefetch the next A/B tiles into it

    drain all outstanding WGMMA

    atomically add acc into the shared FP32 output tile
    old_count = atomically increment this tile's completion counter

    if old_count == SPLIT_K - 1:
        # This is the last arriving split for the tile.
        result = load the complete FP32 accumulator tile
        optionally apply ReLU
        store result to C
        clear the accumulator tile
        release-reset the counter to zero
```

The central mental model is:

```text
TLX overlaps global loads with WGMMA inside each split,
then replaces the explicit partial tensor and second reducer launch
with a per-tile atomic accumulator and last-arrival protocol.
```
