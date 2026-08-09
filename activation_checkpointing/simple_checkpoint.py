"""Small, faithful implementations of PyTorch-style activation checkpointing.

This module deliberately does not call ``torch.utils.checkpoint``.  Its core
control flow is adapted from PyTorch's BSD-licensed implementation in
``torch/utils/checkpoint.py`` (PyTorch 2.11), with the production-only machinery
removed and the teaching-relevant machinery left intact.
"""

from __future__ import annotations

import contextlib
import enum
import weakref
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, TypeAlias

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


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


class _Handle:
    pass


class _Holder:
    def __init__(self) -> None:
        self.handle: _Handle | None = None


class _StopRecomputationError(Exception):
    """Internal control flow: all forward save points have been replayed."""


class _CheckpointFrame:
    def __init__(
        self,
        recompute_fn: Callable[..., None],
        *,
        early_stop: bool,
        determinism_check: bool,
        stats: CheckpointStats,
    ) -> None:
        self.recompute_fn = recompute_fn
        self.early_stop = early_stop
        self.determinism_check = determinism_check
        self.stats = stats
        self.input_saver: torch.Tensor | None = None

        # Autograd owns each Holder; the frame intentionally owns only weak refs.
        self.weak_holders: list[weakref.ReferenceType[_Holder]] = []
        self.forward_metadata: list[TensorMetadata] = []

        # A handle lives exactly as long as its Holder needs the replayed tensor.
        self.recomputed: weakref.WeakKeyDictionary[_Handle, torch.Tensor] = (
            weakref.WeakKeyDictionary()
        )
        self.recompute_counter = 0
        self.is_recomputed = False
        self.forward_completed = False

    def check_recomputed_tensors_match(self) -> None:
        if self.recompute_counter != len(self.weak_holders):
            raise CheckpointError(
                "A different number of tensors was saved during forward and "
                f"recomputation ({len(self.weak_holders)} vs "
                f"{self.recompute_counter}). The checkpointed function must be "
                "the same in both passes."
            )

        if not self.determinism_check:
            return

        mismatches: list[str] = []
        for index, holder_ref in enumerate(self.weak_holders):
            holder = holder_ref()
            if holder is None:
                continue
            handle = holder.handle
            if handle is None or handle not in self.recomputed:
                raise CheckpointError(
                    f"Recomputation did not produce saved tensor {index}."
                )
            actual = _metadata(self.recomputed[handle])
            expected = self.forward_metadata[index]
            if actual != expected:
                mismatches.append(f"#{index}: forward={expected}, replay={actual}")

        if mismatches:
            raise CheckpointError(
                "Recomputed tensors have different shape/dtype/device metadata:\n"
                + "\n".join(mismatches)
            )


class _RecomputationHooks(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame_ref: weakref.ReferenceType[_CheckpointFrame]) -> None:
        def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
            frame = frame_ref()
            if frame is None:
                raise AssertionError("checkpoint frame was released during replay")

            tensor = tensor.detach() if tensor.requires_grad else tensor
            index = frame.recompute_counter
            frame.recompute_counter += 1
            frame.stats.recompute_saved_tensors_seen += 1

            if index >= len(frame.weak_holders):
                raise CheckpointError(
                    "Recomputation saved more tensors than the original forward."
                )

            holder = frame.weak_holders[index]()
            if holder is not None:
                handle = _Handle()
                holder.handle = handle
                frame.recomputed[handle] = tensor

            if frame.early_stop and frame.recompute_counter == len(frame.weak_holders):
                frame.stats.stopped_early = True
                raise _StopRecomputationError
            return tensor

        super().__init__(pack_hook, lambda tensor: tensor)


class _CheckpointHooks(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame: _CheckpointFrame) -> None:
        def pack_hook(tensor: torch.Tensor) -> _Holder:
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            frame.forward_metadata.append(_metadata(tensor))
            frame.stats.forward_saved_tensors_intercepted += 1
            frame.stats.forward_saved_tensor_bytes_intercepted += _logical_tensor_bytes(
                tensor
            )
            # Crucial: autograd saves this tiny Python holder, not the activation.
            return holder

        def unpack_hook(holder: _Holder) -> torch.Tensor:
            if not frame.is_recomputed:
                if frame.input_saver is None or frame.input_saver.grad_fn is None:
                    raise AssertionError("checkpoint inputs were not saved")
                ctx = frame.input_saver.grad_fn
                inputs = ctx.get_args(ctx.saved_tensors)
                frame.stats.recomputations += 1
                try:
                    with _RecomputationHooks(weakref.ref(frame)), torch.enable_grad():
                        frame.recompute_fn(*inputs)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed = True
                frame.check_recomputed_tensors_match()

            handle = holder.handle
            if handle is None or handle not in frame.recomputed:
                raise CheckpointError(
                    "A saved tensor was unpacked twice. This compact implementation "
                    "supports one backward pass per checkpointed graph."
                )
            tensor = frame.recomputed[handle]
            holder.handle = None
            return tensor

        super().__init__(pack_hook, unpack_hook)


class _NoopSaveInputs(torch.autograd.Function):
    """Put checkpoint inputs on the outer graph without running a real op."""

    @staticmethod
    def forward(*args: Any) -> torch.Tensor:
        return torch.empty(0)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        tensor_positions = [
            (index, value)
            for index, value in enumerate(inputs)
            if isinstance(value, torch.Tensor)
        ]
        position_to_saved = {
            position: saved for saved, (position, _) in enumerate(tensor_positions)
        }
        placeholders = [
            None if isinstance(value, torch.Tensor) else value for value in inputs
        ]

        def get_args(saved_tensors: tuple[torch.Tensor, ...]) -> list[Any]:
            restored = [
                saved_tensors[position_to_saved[index]]
                if index in position_to_saved
                else value
                for index, value in enumerate(placeholders)
            ]
            return restored[1:]  # Drop the dummy tensor.

        ctx.get_args = get_args
        ctx.save_for_backward(*(tensor for _, tensor in tensor_positions))

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> None:
        raise AssertionError("the input saver must never itself be differentiated")


def _null_contexts() -> tuple[ContextManager[Any], ContextManager[Any]]:
    return contextlib.nullcontext(), contextlib.nullcontext()


def _device_type_and_indices(tree: Any) -> tuple[str, list[int]]:
    device_type = "cpu"
    indices: list[int] = []

    def visit(value: Any) -> Any:
        nonlocal device_type
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            if device_type not in ("cpu", value.device.type):
                raise ValueError("checkpoint inputs span multiple accelerator types")
            device_type = value.device.type
            if value.device.index is not None and value.device.index not in indices:
                indices.append(value.device.index)
        return value

    tree_map(visit, tree)
    return device_type, indices


def _autocast_kwargs(device_type: str) -> dict[str, Any]:
    return {
        "enabled": torch.is_autocast_enabled(device_type),
        "dtype": torch.get_autocast_dtype(device_type),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }


def _checkpoint_generator(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    preserve_rng_state: bool,
    context_fn: ContextFn,
    determinism_check: bool,
    early_stop: bool,
    stats: CheckpointStats,
):
    forward_context, recompute_context = context_fn()
    device_type, device_indices = _device_type_and_indices((args, kwargs))

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

    frame = _CheckpointFrame(
        recompute_fn,
        early_stop=early_stop,
        determinism_check=determinism_check,
        stats=stats,
    )
    dummy = torch.empty(0, requires_grad=True)
    frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, *args)

    if frame.input_saver.grad_fn is None:
        yield
        return

    with _CheckpointHooks(frame), forward_context:
        yield
    frame.forward_completed = True

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
    """Run ``function`` using PyTorch's non-reentrant checkpointing mechanism.

    Unlike ``torch.utils.checkpoint.checkpoint``, this teaching version supports
    one backward traversal and eager execution only. Keyword arguments, dropout
    RNG replay, autocast state, early stopping, and metadata checks are retained.
    """

    stats = CheckpointStats() if stats is None else stats
    generator = _checkpoint_generator(
        function,
        args,
        kwargs,
        preserve_rng_state=preserve_rng_state,
        context_fn=context_fn,
        determinism_check=determinism_check,
        early_stop=early_stop,
        stats=stats,
    )
    next(generator)
    result = function(*args, **kwargs)
    try:
        next(generator)
    except StopIteration:
        return result
    raise AssertionError("checkpoint generator did not finish")


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


class _VersionWrapper:
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


class _CachingTorchDispatchMode(TorchDispatchMode):
    def __init__(
        self,
        policy_fn: Callable[..., CheckpointPolicy | bool],
        storage: defaultdict[Any, list[Any]],
        stats: SelectiveCheckpointStats,
    ) -> None:
        self.policy_fn = policy_fn
        self.storage = storage
        self.stats = stats

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in _SAC_IGNORED_OPS:
            return func(*args, **({} if kwargs is None else kwargs))

        kwargs = {} if kwargs is None else kwargs
        policy = _normalize_policy(
            self.policy_fn(
                SelectiveCheckpointContext(is_recompute=False),
                func,
                *args,
                **kwargs,
            )
        )
        name = str(func)
        self.stats.forward_ops[name] += 1
        output = func(*args, **kwargs)

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE):
            output_may_alias = any(
                result.alias_info is not None for result in func._schema.returns
            )
            cached = tree_map(
                lambda value: _VersionWrapper(
                    _maybe_detach(value, output_may_alias)
                ),
                output,
            )
            self.storage[func].append(cached)
            self.stats.cached_ops[name] += 1
            self.stats.cached_tensor_bytes += sum(
                _logical_tensor_bytes(leaf)
                for leaf in _tree_leaves_without_import(output)
            )
        return output


class _CachedTorchDispatchMode(TorchDispatchMode):
    def __init__(
        self,
        policy_fn: Callable[..., CheckpointPolicy | bool],
        storage: defaultdict[Any, list[Any]],
        stats: SelectiveCheckpointStats,
        allow_cache_entry_mutation: bool,
    ) -> None:
        self.policy_fn = policy_fn
        self.storage = storage
        self.stats = stats
        self.allow_cache_entry_mutation = allow_cache_entry_mutation

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in _SAC_IGNORED_OPS:
            return func(*args, **({} if kwargs is None else kwargs))

        kwargs = {} if kwargs is None else kwargs
        policy = _normalize_policy(
            self.policy_fn(
                SelectiveCheckpointContext(is_recompute=True),
                func,
                *args,
                **kwargs,
            )
        )
        name = str(func)
        self.stats.recompute_ops[name] += 1

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE):
            cached_values = self.storage.get(func)
            if not cached_values:
                raise RuntimeError(
                    f"{func} was marked SAVE during replay but has no forward cache "
                    "entry. The policy or execution order changed."
                )
            self.stats.reused_ops[name] += 1
            return tree_map(
                lambda wrapper: wrapper.get_value(
                    self.allow_cache_entry_mutation
                ),
                cached_values.pop(0),
            )
        return func(*args, **kwargs)


def _tree_leaves_without_import(value: Any) -> list[Any]:
    leaves: list[Any] = []

    def collect(leaf: Any) -> Any:
        leaves.append(leaf)
        return leaf

    tree_map(collect, value)
    return leaves


def create_selective_checkpoint_contexts(
    policy_fn_or_ops: Callable[..., CheckpointPolicy | bool] | list[Any],
    *,
    allow_cache_entry_mutation: bool = False,
    stats: SelectiveCheckpointStats | None = None,
) -> tuple[_CachingTorchDispatchMode, _CachedTorchDispatchMode]:
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

    storage: defaultdict[Any, list[Any]] = defaultdict(list)
    stats = SelectiveCheckpointStats() if stats is None else stats
    return (
        _CachingTorchDispatchMode(policy_fn, storage, stats),
        _CachedTorchDispatchMode(
            policy_fn, storage, stats, allow_cache_entry_mutation
        ),
    )
