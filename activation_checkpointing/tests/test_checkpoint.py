from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from decoder import DecoderBlock, DecoderConfig
from simple_checkpoint import (
    CheckpointError,
    CheckpointStats,
    SelectiveCheckpointStats,
    checkpoint,
    create_selective_checkpoint_contexts,
)


def _run_decoder(model, x, wrapper):
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    torch.manual_seed(123)
    output = wrapper(model, x)
    output.float().square().mean().backward()
    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert x.grad is not None
    return output.detach(), x.grad.detach(), parameter_gradients


def _assert_run_close(actual, expected, *, atol=1e-6, rtol=1e-5):
    actual_output, actual_input_grad, actual_parameter_grads = actual
    expected_output, expected_input_grad, expected_parameter_grads = expected
    torch.testing.assert_close(actual_output, expected_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        actual_input_grad, expected_input_grad, atol=atol, rtol=rtol
    )
    assert actual_parameter_grads.keys() == expected_parameter_grads.keys()
    for name in actual_parameter_grads:
        torch.testing.assert_close(
            actual_parameter_grads[name],
            expected_parameter_grads[name],
            atol=atol,
            rtol=rtol,
        )


def _small_decoder(dropout=0.1):
    return DecoderBlock(
        DecoderConfig(dim=64, num_heads=4, hidden_dim=128, dropout=dropout)
    ).train()


def test_decoder_outputs_and_all_gradients_match_eager_with_dropout():
    torch.manual_seed(0)
    eager_model = _small_decoder()
    checkpointed_model = copy.deepcopy(eager_model)
    x = torch.randn(2, 12, 64)

    eager = _run_decoder(eager_model, x, lambda model, value: model(value))
    actual = _run_decoder(
        checkpointed_model,
        x,
        lambda model, value: checkpoint(model, value),
    )

    _assert_run_close(actual, eager)


def test_decoder_matches_pytorch_non_reentrant_checkpoint():
    torch.manual_seed(0)
    native_model = _small_decoder()
    standalone_model = copy.deepcopy(native_model)
    x = torch.randn(1, 8, 64)

    native = _run_decoder(
        native_model,
        x,
        lambda model, value: torch_checkpoint(model, value, use_reentrant=False),
    )
    standalone = _run_decoder(
        standalone_model,
        x,
        lambda model, value: checkpoint(model, value),
    )

    _assert_run_close(standalone, native)


def test_rng_after_backward_matches_eager_execution():
    model = _small_decoder()
    checkpointed_model = copy.deepcopy(model)
    x = torch.randn(1, 8, 64)

    torch.manual_seed(99)
    eager_x = x.clone().requires_grad_(True)
    model(eager_x).sum().backward()
    eager_rng_after_backward = torch.get_rng_state()

    torch.manual_seed(99)
    checkpoint_x = x.clone().requires_grad_(True)
    checkpoint(checkpointed_model, checkpoint_x).sum().backward()
    checkpoint_rng_after_backward = torch.get_rng_state()

    assert torch.equal(checkpoint_rng_after_backward, eager_rng_after_backward)


def test_early_stop_interrupts_function_after_last_saved_tensor():
    early_events = []
    full_events = []

    def early_fn(x):
        result = x.sin().cos()
        early_events.append("tail")
        return result

    def full_fn(x):
        result = x.sin().cos()
        full_events.append("tail")
        return result

    early_x = torch.randn(8, requires_grad=True)
    full_x = early_x.detach().clone().requires_grad_(True)
    early_stats = CheckpointStats()
    full_stats = CheckpointStats()
    checkpoint(early_fn, early_x, early_stop=True, stats=early_stats).sum().backward()
    checkpoint(full_fn, full_x, early_stop=False, stats=full_stats).sum().backward()

    assert early_events == ["tail"]
    assert full_events == ["tail", "tail"]
    assert early_stats.stopped_early
    assert not full_stats.stopped_early
    torch.testing.assert_close(early_x.grad, full_x.grad)


def test_selective_checkpoint_reuses_saved_mm_and_matches_gradients():
    torch.manual_seed(0)
    x = torch.randn(8, 16, requires_grad=True)
    weight = torch.randn(16, 16, requires_grad=True)
    eager_x = x.detach().clone().requires_grad_(True)
    eager_weight = weight.detach().clone().requires_grad_(True)

    eager = torch.sigmoid(eager_x @ eager_weight)
    eager.sum().backward()

    stats = SelectiveCheckpointStats()
    contexts = create_selective_checkpoint_contexts(
        [torch.ops.aten.mm.default], stats=stats
    )
    actual = checkpoint(
        lambda left, right: torch.sigmoid(left @ right),
        x,
        weight,
        context_fn=lambda: contexts,
    )
    actual.sum().backward()

    torch.testing.assert_close(actual, eager)
    torch.testing.assert_close(x.grad, eager_x.grad)
    torch.testing.assert_close(weight.grad, eager_weight.grad)
    assert stats.cached_ops["aten.mm.default"] == 1
    assert stats.reused_ops["aten.mm.default"] == 1


def test_selective_checkpoint_detects_mutated_cached_tensor():
    def mutating_function(x):
        intermediate = x * 2
        intermediate.add_(1)
        return intermediate.square()

    def policy(ctx, op, *args, **kwargs):
        del ctx, args, kwargs
        return op == torch.ops.aten.mul.Tensor

    contexts = create_selective_checkpoint_contexts(policy)
    x = torch.randn(8, requires_grad=True)
    output = checkpoint(mutating_function, x, context_fn=lambda: contexts)

    with pytest.raises(RuntimeError, match="mutated in-place"):
        output.sum().backward()


def test_determinism_check_reports_changed_recompute_shape():
    state = {"shorten": False}

    def changing_function(x):
        intermediate = x.sin()
        if state["shorten"]:
            intermediate = intermediate[:2]
        return intermediate.cos()

    x = torch.randn(8, requires_grad=True)
    output = checkpoint(changing_function, x)
    state["shorten"] = True

    with pytest.raises(CheckpointError, match="different shape/dtype/device"):
        output.sum().backward()


def test_no_grad_bypasses_checkpoint_hooks():
    stats = CheckpointStats()
    with torch.no_grad():
        output = checkpoint(lambda x: x.sin(), torch.randn(4), stats=stats)

    assert output.grad_fn is None
    assert stats.forward_saved_tensors_intercepted == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_decoder_smoke_matches_eager():
    torch.manual_seed(0)
    eager_model = _small_decoder().to(device="cuda", dtype=torch.bfloat16)
    checkpointed_model = copy.deepcopy(eager_model)
    x = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16)

    eager = _run_decoder(eager_model, x, lambda model, value: model(value))
    actual = _run_decoder(
        checkpointed_model,
        x,
        lambda model, value: checkpoint(model, value),
    )

    _assert_run_close(actual, eager, atol=2e-3, rtol=2e-2)


def test_keyword_inputs_and_parameter_only_gradients():
    weight = torch.randn(4, 4, requires_grad=True)
    eager_weight = weight.detach().clone().requires_grad_(True)
    x = torch.randn(3, 4)
    expected = (x @ eager_weight).sin() * 2
    expected.sum().backward()

    actual = checkpoint(lambda value, *, scale: (value @ weight).sin() * scale, x, scale=2)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weight.grad, eager_weight.grad)


def test_forward_activations_are_freed_before_backward():
    import weakref

    activations = []

    def function(x):
        intermediate = x.sin()
        activations.append(weakref.ref(intermediate))
        return intermediate.cos()

    x = torch.randn(8, requires_grad=True)
    output = checkpoint(function, x)
    assert activations[0]() is None
    output.sum().backward()
    assert all(ref() is None for ref in activations)
    torch.testing.assert_close(x.grad, -x.sin().sin() * x.cos())


def test_forward_exception_exits_context_and_restores_hooks():
    import contextlib

    events = []

    @contextlib.contextmanager
    def forward_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def failing_function(x):
        x.sin()
        raise ValueError("forward failed")

    x = torch.randn(8, requires_grad=True)
    stats = CheckpointStats()
    # Keep the exception traceback alive: cleanup must not depend on GC.
    with pytest.raises(ValueError, match="forward failed") as error:
        checkpoint(
            failing_function, x, stats=stats,
            context_fn=lambda: (forward_context(), contextlib.nullcontext()),
        )
    assert error.value is not None
    assert events == ["enter", "exit"]
    intercepted = stats.forward_saved_tensors_intercepted
    x.sin().sum().backward()
    assert stats.forward_saved_tensors_intercepted == intercepted
    torch.testing.assert_close(x.grad, x.cos())


@pytest.mark.parametrize("early_stop", [True, False])
def test_selective_repeated_ops_match_native_and_preserve_fifo(early_stop):
    from torch.utils.checkpoint import create_selective_checkpoint_contexts as native_contexts

    def function(x, weight):
        return ((x @ weight).sin() @ weight).sigmoid()

    torch.manual_seed(4)
    x = torch.randn(4, 4, requires_grad=True)
    weight = torch.randn(4, 4, requires_grad=True)
    native_x = x.detach().clone().requires_grad_(True)
    native_weight = weight.detach().clone().requires_grad_(True)
    ops = [torch.ops.aten.mm.default]
    expected = torch_checkpoint(
        function, native_x, native_weight, use_reentrant=False,
        context_fn=lambda: native_contexts(ops), early_stop=early_stop,
    )
    expected.sum().backward()

    stats = SelectiveCheckpointStats()
    actual = checkpoint(
        function, x, weight, early_stop=early_stop,
        context_fn=lambda: create_selective_checkpoint_contexts(ops, stats=stats),
    )
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, native_x.grad)
    torch.testing.assert_close(weight.grad, native_weight.grad)
    assert stats.cached_ops["aten.mm.default"] == 2
    assert stats.reused_ops["aten.mm.default"] == 2


@pytest.mark.parametrize("selective", [False, True])
def test_cpu_autocast_is_restored_for_replay(selective):
    x = torch.randn(4, 4, requires_grad=True)
    eager_x = x.detach().clone().requires_grad_(True)
    options = {}
    if selective:
        options["context_fn"] = lambda: create_selective_checkpoint_contexts(
            [torch.ops.aten.mm.default]
        )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = (eager_x @ eager_x).sin()
        actual = checkpoint(lambda value: (value @ value).sin(), x, **options)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, eager_x.grad)


@pytest.mark.parametrize("extra_save", [False, True])
def test_changed_save_count_is_rejected(extra_save):
    replay = False

    def function(x):
        result = x.sin()
        if replay == extra_save:
            result = result.cos()
        return result

    output = checkpoint(function, torch.randn(4, requires_grad=True), early_stop=False)
    replay = True
    with pytest.raises(CheckpointError, match="more tensors|different number"):
        output.sum().backward()


def test_boundary_input_mutation_still_uses_autograd_version_check():
    x = torch.randn(4, requires_grad=True)
    output = checkpoint(lambda value: value.sin(), x)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()
