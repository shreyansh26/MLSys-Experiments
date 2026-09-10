"""Activation checkpointing as save slots, replay hooks, and an optional op cache.

This module deliberately does not call ``torch.utils.checkpoint``.  Its core
control flow is adapted from PyTorch's BSD-licensed implementation in
``torch/utils/checkpoint.py`` (PyTorch 2.11), with the production-only machinery
removed and the teaching-relevant machinery left intact.
"""

from __future__ import annotations

import contextlib
import enum
import weakref
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, TypeAlias

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves, tree_map


TensorMetadata: TypeAlias = tuple[torch.Size, torch.dtype, torch.device]
ContextFn: TypeAlias = Callable[[], tuple[ContextManager[Any], ContextManager[Any]]]


class CheckpointError(RuntimeError):
    """The recomputation no longer matches the original forward pass."""


@dataclass
class CheckpointStats:
    """Instrumentation that is intentionally absent from PyTorch's public API."""

    recomputations: int = 0
    forward_saved_tensors_intercepted: int = 0
    forward_saved_tensor_bytes_intercepted: int = 0
    recompute_saved_tensors_seen: int = 0
    stopped_early: bool = False


def _logical_tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return value.numel() * value.element_size()


def _metadata(tensor: torch.Tensor) -> TensorMetadata:
    return tensor.shape, tensor.dtype, tensor.device


class _SavedActivation:
    """An empty slot in forward; replay fills it when backward needs it."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.metadata = _metadata(tensor)
        self.tensor: torch.Tensor | None = None


class _StopRecomputation(Exception):
    """All forward save points have been replayed; skip the function's tail."""


class _Checkpoint:
    """The three hooks that replace, rebuild, and retrieve saved activations."""

    def __init__(
        self,
        replay: Callable[..., None],
        inputs: torch.Tensor,
        early_stop: bool,
        determinism_check: bool,
        stats: CheckpointStats,
    ) -> None:
        self.replay = replay
        self.inputs = inputs
        self.early_stop = early_stop
        self.determinism_check = determinism_check
        self.stats = stats
        # Autograd owns the slots. Weak refs let unused graph branches free them.
        self.slots: list[weakref.ReferenceType[_SavedActivation]] = []
        self.replay_count = 0
        self.replayed = False

    def pack(self, tensor: torch.Tensor) -> _SavedActivation:
        slot = _SavedActivation(tensor)
        self.slots.append(weakref.ref(slot))
        self.stats.forward_saved_tensors_intercepted += 1
        self.stats.forward_saved_tensor_bytes_intercepted += _logical_tensor_bytes(tensor)
        return slot  # Save a slot in the graph, not the activation.

    def pack_recomputed(self, tensor: torch.Tensor) -> torch.Tensor:
        index = self.replay_count
        self.replay_count += 1
        self.stats.recompute_saved_tensors_seen += 1
        if index >= len(self.slots):
            raise CheckpointError(
                "Recomputation saved more tensors than the original forward."
            )
        tensor = tensor.detach() if tensor.requires_grad else tensor
        slot = self.slots[index]()
        if slot is not None:
            slot.tensor = tensor
        if self.early_stop and self.replay_count == len(self.slots):
            self.stats.stopped_early = True
            raise _StopRecomputation
        return tensor

    def unpack(self, slot: _SavedActivation) -> torch.Tensor:
        if not self.replayed:
            ctx = self.inputs.grad_fn
            inputs = ctx.get_args(ctx.saved_tensors)
            self.stats.recomputations += 1
            try:
                with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(
                    self.pack_recomputed, lambda tensor: tensor
                ):
                    self.replay(*inputs)
            except _StopRecomputation:
                pass
            self.replayed = True
            self.check_replay()

        if slot.tensor is None:
            raise CheckpointError(
                "A saved tensor was unpacked twice. This compact implementation "
                "supports one backward pass per checkpointed graph."
            )
        tensor = slot.tensor
        slot.tensor = None  # Release each activation as backward consumes it.
        return tensor

    def check_replay(self) -> None:
        if self.replay_count != len(self.slots):
            raise CheckpointError(
                "A different number of tensors was saved during forward and "
                f"recomputation ({len(self.slots)} vs {self.replay_count}). "
                "The checkpointed function must be the same in both passes."
            )
        if not self.determinism_check:
            return
        mismatches = []
        for index, slot_ref in enumerate(self.slots):
            slot = slot_ref()
            if slot is None:
                continue
            if slot.tensor is None:
                raise CheckpointError(f"Recomputation did not produce saved tensor {index}.")
            actual = _metadata(slot.tensor)
            if actual != slot.metadata:
                mismatches.append(f"#{index}: forward={slot.metadata}, replay={actual}")
        if mismatches:
            raise CheckpointError(
                "Recomputed tensors have different shape/dtype/device metadata:\n"
                + "\n".join(mismatches)
            )


class _SaveInputs(torch.autograd.Function):
    """Put checkpoint inputs on the outer graph without running a real op."""

    @staticmethod
    def forward(*args: Any) -> torch.Tensor:
        return torch.empty(0)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        # Keep non-tensors in Python and let autograd own the tensor inputs.
        positions = [
            i for i, value in enumerate(inputs) if isinstance(value, torch.Tensor)
        ]
        arguments = [
            None if isinstance(value, torch.Tensor) else value for value in inputs
        ]

        def get_args(saved_tensors: tuple[torch.Tensor, ...]) -> list[Any]:
            restored = arguments.copy()
            for index, tensor in zip(positions, saved_tensors, strict=True):
                restored[index] = tensor
            return restored[1:]  # Drop the dummy tensor.

        ctx.get_args = get_args
        ctx.save_for_backward(*(inputs[i] for i in positions))

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> None:
        raise AssertionError("the input saver must never itself be differentiated")


def _null_contexts() -> tuple[ContextManager[Any], ContextManager[Any]]:
    return contextlib.nullcontext(), contextlib.nullcontext()


def _device_type_and_indices(tree: Any) -> tuple[str, list[int]]:
    device_type = "cpu"
    indices: list[int] = []

    for value in tree_leaves(tree):
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            if device_type not in ("cpu", value.device.type):
                raise ValueError("checkpoint inputs span multiple accelerator types")
            device_type = value.device.type
            if value.device.index is not None and value.device.index not in indices:
                indices.append(value.device.index)
    return device_type, indices


def _autocast_kwargs(device_type: str) -> dict[str, Any]:
    return {
        "enabled": torch.is_autocast_enabled(device_type),
        "dtype": torch.get_autocast_dtype(device_type),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }


def checkpoint(
    function: Callable[..., Any],
    *args: Any,
    preserve_rng_state: bool = True,
    context_fn: ContextFn = _null_contexts,
    determinism_check: bool = True,
    early_stop: bool = True,
    stats: CheckpointStats | None = None,
    **kwargs: Any,
) -> Any:
    """Run forward with empty activation slots, then fill them on first backward.

    Supports eager execution and one backward traversal. Keyword arguments,
    dropout RNG replay, autocast state, early stopping, and metadata checks are
    retained. This implementation never calls ``torch.utils.checkpoint``.
    """
    stats = CheckpointStats() if stats is None else stats
    forward_context, recompute_context = context_fn()
    device_type, device_indices = _device_type_and_indices((args, kwargs))

    # Capture the execution environment once, before forward changes it.
    cpu_rng_state = torch.get_rng_state() if preserve_rng_state else None
    device_module = getattr(torch, device_type, None)
    device_was_initialized = bool(
        device_type != "cpu" and getattr(device_module, "_initialized", False)
    )
    device_rng_states: list[torch.Tensor] = []
    if preserve_rng_state and device_was_initialized:
        for index in device_indices:
            with device_module.device(index):
                device_rng_states.append(device_module.get_rng_state())

    device_amp = _autocast_kwargs(device_type)
    cpu_amp = _autocast_kwargs("cpu")

    def recompute_fn(saved_kwargs: dict[str, Any], *saved_args: Any) -> None:
        replay_devices = device_indices if device_was_initialized else []
        with torch.random.fork_rng(
            devices=replay_devices,
            enabled=preserve_rng_state,
            device_type=device_type,
        ):
            if preserve_rng_state:
                assert cpu_rng_state is not None
                torch.set_rng_state(cpu_rng_state)
                if device_was_initialized:
                    for index, state in zip(
                        device_indices, device_rng_states, strict=True
                    ):
                        with device_module.device(index):
                            device_module.set_rng_state(state)

            with contextlib.ExitStack() as stack:
                stack.enter_context(torch.amp.autocast(device_type, **device_amp))
                if device_type != "cpu":
                    stack.enter_context(torch.amp.autocast("cpu", **cpu_amp))
                stack.enter_context(recompute_context)
                function(*saved_args, **saved_kwargs)

    # The dummy creates an autograd node even when only model parameters need
    # gradients. Saving inputs on that node preserves autograd's version checks.
    inputs = _SaveInputs.apply(torch.empty(0, requires_grad=True), kwargs, *args)
    if inputs.grad_fn is None:
        return function(*args, **kwargs)

    state = _Checkpoint(recompute_fn, inputs, early_stop, determinism_check, stats)
    with torch.autograd.graph.saved_tensors_hooks(state.pack, state.unpack):
        with forward_context:
            result = function(*args, **kwargs)

    if (
        preserve_rng_state
        and device_type != "cpu"
        and not device_was_initialized
        and getattr(device_module, "_initialized", False)
    ):
        raise RuntimeError(
            "The accelerator was initialized inside the checkpointed forward, "
            "so its pre-forward RNG state could not be captured."
        )

    return result


class CheckpointPolicy(enum.Enum):
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3


@dataclass(frozen=True)
class SelectiveCheckpointContext:
    is_recompute: bool


@dataclass
class SelectiveCheckpointStats:
    forward_ops: Counter[str] = field(default_factory=Counter)
    recompute_ops: Counter[str] = field(default_factory=Counter)
    cached_ops: Counter[str] = field(default_factory=Counter)
    reused_ops: Counter[str] = field(default_factory=Counter)
    cached_tensor_bytes: int = 0


class _CachedValue:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.version = value._version if isinstance(value, torch.Tensor) else None

    def get_value(self, allow_mutation: bool) -> Any:
        if (
            self.version is not None
            and not allow_mutation
            and self.value._version != self.version
        ):
            raise RuntimeError(
                "A tensor cached by selective checkpointing was mutated in-place."
            )
        return self.value


_SAC_IGNORED_OPS = {
    torch.ops.aten.detach.default,
    torch.ops.aten.dim.default,
    torch.ops.aten.is_contiguous.default,
    torch.ops.aten.is_contiguous.memory_format,
    torch.ops.aten.is_non_overlapping_and_dense.default,
    torch.ops.aten.numel.default,
    torch.ops.aten.size.default,
    torch.ops.aten.storage_offset.default,
    torch.ops.aten.stride.default,
    torch.ops.aten.sym_numel.default,
    torch.ops.aten.sym_size.default,
    torch.ops.aten.sym_storage_offset.default,
    torch.ops.aten.sym_stride.default,
}


def _normalize_policy(value: CheckpointPolicy | bool) -> CheckpointPolicy:
    if isinstance(value, bool):
        return (
            CheckpointPolicy.MUST_SAVE
            if value
            else CheckpointPolicy.PREFER_RECOMPUTE
        )
    if not isinstance(value, CheckpointPolicy):
        raise TypeError("selective checkpoint policy must return CheckpointPolicy")
    return value


def _maybe_detach(value: Any, output_may_alias: bool) -> Any:
    if isinstance(value, torch.Tensor) and (
        value.is_floating_point() or value.is_complex() or output_may_alias
    ):
        # Match PyTorch's important alias/version-counter behavior.
        with torch._C._SetExcludeDispatchKeyGuard(
            torch._C.DispatchKey.ADInplaceOrView, False
        ):
            return value.detach()
    return value


class _SelectiveMode(TorchDispatchMode):
    """Forward appends selected outputs; replay consumes them in the same order."""

    def __init__(self, policy_fn, cache, stats, *, is_recompute, allow_mutation=False):
        super().__init__()
        self.policy_fn = policy_fn
        self.cache = cache
        self.stats = stats
        self.context = SelectiveCheckpointContext(is_recompute=is_recompute)
        self.allow_mutation = allow_mutation

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func in _SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        policy = _normalize_policy(self.policy_fn(self.context, func, *args, **kwargs))
        save = policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)
        replay = self.context.is_recompute
        name = str(func)
        counts = self.stats.recompute_ops if replay else self.stats.forward_ops
        counts[name] += 1

        if replay and save:
            entries = self.cache.get(func)
            if not entries:
                raise RuntimeError(
                    f"{func} was marked SAVE during replay but has no forward cache "
                    "entry. The policy or execution order changed."
                )
            self.stats.reused_ops[name] += 1
            return tree_map(
                lambda entry: entry.get_value(self.allow_mutation), entries.popleft()
            )

        output = func(*args, **kwargs)
        if save and not replay:
            output_may_alias = any(
                result.alias_info is not None for result in func._schema.returns
            )
            cached = tree_map(
                lambda value: _CachedValue(_maybe_detach(value, output_may_alias)),
                output,
            )
            self.cache[func].append(cached)
            self.stats.cached_ops[name] += 1
            self.stats.cached_tensor_bytes += sum(
                _logical_tensor_bytes(leaf) for leaf in tree_leaves(output)
            )
        return output


def create_selective_checkpoint_contexts(
    policy_fn_or_ops: Callable[..., CheckpointPolicy | bool] | list[Any],
    *,
    allow_cache_entry_mutation: bool = False,
    stats: SelectiveCheckpointStats | None = None,
) -> tuple[_SelectiveMode, _SelectiveMode]:
    """Create the paired forward-cache and backward-replay dispatch modes.

    The modes share FIFO storage keyed by ``OpOverload``, exactly as PyTorch's
    eager selective-checkpoint implementation does.
    """

    if isinstance(policy_fn_or_ops, list):
        for op in policy_fn_or_ops:
            if not isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
                raise ValueError(
                    "operation lists require a concrete overload, for example "
                    "torch.ops.aten.mm.default"
                )
        selected = frozenset(policy_fn_or_ops)

        def policy_fn(ctx, op, *args, **kwargs):
            del ctx, args, kwargs
            return (
                CheckpointPolicy.MUST_SAVE
                if op in selected
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

    elif callable(policy_fn_or_ops):
        policy_fn = policy_fn_or_ops
    else:
        raise TypeError("policy must be a callable or a list of OpOverloads")

    cache: defaultdict[Any, deque[Any]] = defaultdict(deque)
    stats = SelectiveCheckpointStats() if stats is None else stats
    return (
        _SelectiveMode(policy_fn, cache, stats, is_recompute=False),
        _SelectiveMode(
            policy_fn, cache, stats, is_recompute=True,
            allow_mutation=allow_cache_entry_mutation,
        ),
    )
