"""Load TLX with the exact extension-enabled Triton ABI it was built for."""

from __future__ import annotations

import os
import sysconfig
from types import ModuleType


def _configure_plugin() -> None:
    if "TRITON_PLUGIN_PATHS" in os.environ:
        return
    plugin = os.path.join(
        sysconfig.get_paths()["purelib"], "utlx_plugin", "libutlx.so"
    )
    if not os.path.isfile(plugin):
        raise RuntimeError(f"TLX plugin not found at {plugin}; run `uv sync`")
    os.environ["TRITON_PLUGIN_PATHS"] = plugin


def _install_frontend_compat() -> None:
    """Supply frontend helpers expected by the TLX 3.7 plugin DSL."""
    import triton.language as tl
    import triton.language.core as tl_core
    import triton.language.semantic as triton_semantic
    from triton import knobs
    from triton._C.libtriton import ir

    # The 3.7 plugin registers native ops as `utlx_*`, while its Python DSL
    # calls the historical `create_utlx_*` spellings. Both refer to the exact
    # same pybind functions, so expose the legacy aliases once at import time.
    for native_name in dir(ir.builder):
        if not native_name.startswith("utlx_"):
            continue
        legacy_name = f"create_{native_name}"
        if not hasattr(ir.builder, legacy_name):
            setattr(ir.builder, legacy_name, getattr(ir.builder, native_name))

    if not hasattr(tl, "_unwrap_if_constexpr"):
        tl._unwrap_if_constexpr = tl_core._unwrap_if_constexpr

    if not hasattr(triton_semantic.TritonSemantic, "_prepare_legacy_load"):

        def _prepare_legacy_load(
            self, ptr, mask, other, boundary_check, padding
        ):
            if not ptr.type.scalar.is_ptr():
                raise ValueError(f"unsupported pointer type: {ptr.type!r}")
            if mask is None and other is not None:
                raise ValueError("other requires a mask")
            if padding or boundary_check:
                raise ValueError(
                    "boundary_check and padding are unsupported for tensor pointers"
                )
            if not ptr.type.is_block():
                if mask and mask.type.is_block():
                    raise ValueError("block mask with scalar pointer")
                if other and other.type.is_block():
                    raise ValueError("block other with scalar pointer")
            if ptr.type.is_block():
                if mask is not None:
                    ptr, mask = self.broadcast_impl_value(ptr, mask)
                if other is not None:
                    ptr, other = self.broadcast_impl_value(ptr, other)
            ptr_ty = ptr.type.scalar
            element_ty = ptr_ty.element_ty
            is_bool = element_ty == tl.int1
            if is_bool:
                element_ty = tl.int8
                ptr = self.cast(
                    ptr, tl.pointer_type(element_ty, ptr_ty.address_space)
                )
            if other is not None:
                other = self.cast(other, element_ty)
            if ptr.type.is_block():
                dst_ty = tl.block_type(element_ty, ptr.type.get_block_shapes())
            else:
                dst_ty = element_ty
            return dst_ty, ptr, mask, other, is_bool

        triton_semantic.TritonSemantic._prepare_legacy_load = _prepare_legacy_load

    if not hasattr(triton_semantic.TritonSemantic, "dot_precheck"):

        def dot_precheck(
            self,
            lhs,
            rhs,
            acc,
            input_precision,
            allow_tf32,
            max_num_imprecise_acc,
            out_dtype,
            tlx_paired_ctas=False,
        ):
            del tlx_paired_ctas
            input_precision = tl_core._unwrap_if_constexpr(input_precision)
            allow_tf32 = tl_core._unwrap_if_constexpr(allow_tf32)
            out_dtype = tl_core._unwrap_if_constexpr(out_dtype)
            acc = tl_core._unwrap_if_constexpr(acc)
            max_num_imprecise_acc = tl_core._unwrap_if_constexpr(
                max_num_imprecise_acc
            )
            if input_precision is not None and allow_tf32 is not None:
                raise ValueError("set only one of input_precision and allow_tf32")
            if input_precision is None:
                supports_tf32 = (
                    "tf32" in self.builder.options.allowed_dot_input_precisions
                )
                input_precision = knobs.language.fp32_default or (
                    "tf32" if supports_tf32 and allow_tf32 is not False else "ieee"
                )
            if not lhs.type.is_block() or not rhs.type.is_block():
                raise ValueError("dot operands must be blocks")
            if lhs.dtype != rhs.dtype:
                raise ValueError("dot operands must have the same dtype")
            if lhs.dtype not in (
                tl.int8,
                tl.uint8,
                tl.float16,
                tl.bfloat16,
                tl.float32,
                tl.float64,
            ):
                raise ValueError(f"unsupported dot dtype: {lhs.dtype}")
            input_precision = self._str_to_dot_input_precision(input_precision)
            if len(lhs.shape) != len(rhs.shape) or len(lhs.shape) not in (2, 3):
                raise ValueError("dot operands must both be rank 2 or rank 3")
            if lhs.shape[-1] != rhs.shape[-2]:
                raise ValueError("incompatible dot reduction dimensions")
            min_m, min_n, min_k = self.builder.codegen_fns["min_dot_size"](
                lhs.type, rhs.type
            )
            if (
                lhs.shape[-2] < min_m
                or rhs.shape[-1] < min_n
                or lhs.shape[-1] < min_k
            ):
                raise ValueError(
                    f"dot shape must satisfy M>={min_m}, N>={min_n}, K>={min_k}"
                )
            if lhs.type.scalar.is_int():
                zero = self.builder.get_int32(0)
                result_scalar_ty = tl.int32
            elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
                zero = self.builder.get_fp32(0)
                result_scalar_ty = tl.float32
            elif lhs.type.scalar.is_fp64():
                zero = self.builder.get_fp64(0)
                result_scalar_ty = tl.float64
            else:
                zero = (
                    self.builder.get_fp16(0)
                    if out_dtype.is_fp16()
                    else self.builder.get_fp32(0)
                )
                result_scalar_ty = out_dtype
            m, n = lhs.type.shape[-2], rhs.type.shape[-1]
            batch = lhs.type.shape[0] if len(lhs.shape) == 3 else None
            result_ty = tl.block_type(
                result_scalar_ty, [batch, m, n] if batch else [m, n]
            )
            if acc is None:
                acc_handle = self.builder.create_splat(
                    result_ty.to_ir(self.builder), zero
                )
            else:
                if acc.type != result_ty:
                    raise ValueError("incompatible accumulator type")
                acc_handle = acc.handle
            if max_num_imprecise_acc is None:
                max_num_imprecise_acc = 0
            return (
                lhs,
                rhs,
                acc_handle,
                input_precision,
                max_num_imprecise_acc,
                result_ty,
            )

        triton_semantic.TritonSemantic.dot_precheck = dot_precheck


def load_tlx() -> ModuleType:
    _configure_plugin()
    _install_frontend_compat()
    import utlx_plugin as tlx

    # TLX's stage hook predates Gluon and otherwise replaces a non-existent
    # `ttir` stage in Gluon's bytes -> TTGIR pipeline. Leave Gluon on the native
    # NVIDIABackend stages; TLX's extension passes are only meaningful for the
    # stock Triton frontend used by TLX kernels.
    from triton import knobs
    from triton.backends.compiler import Language
    from utlx_plugin import custom_stages

    if not getattr(custom_stages, "_gluon_stage_compatible", False):
        original_stage_hook = custom_stages.inspect_stages_hook

        def _stage_hook(
            self=None,
            stages=None,
            options=None,
            language=None,
            capability=None,
        ):
            if language == Language.GLUON:
                return custom_stages.get_key(), custom_stages.get_hash()
            return original_stage_hook(
                self, stages, options, language, capability
            )

        knobs.runtime.add_stages_inspection_hook = _stage_hook
        custom_stages._gluon_stage_compatible = True

    # A layout-bearing host descriptor is parsed as Gluon's descriptor value,
    # even when stock Triton's frontend compiles the kernel. Both wrappers own
    # the same !tt.tensordesc IR value; TLX 3.7.1's Python guard simply predates
    # the Gluon wrapper. Keep the native op and accept either descriptor value.
    if not getattr(tlx, "_layout_descriptor_compatible", False):
        import triton.language.core as triton_language
        import utlx_plugin.mem_ops as mem_ops
        from utlx_plugin.mma_ops import require_nv_mma_shared_layout

        @triton_language.builtin
        def _async_descriptor_load(
            desc,
            result,
            offsets,
            barrier,
            pred=None,
            cache_modifier="",
            eviction_policy="",
            multicast_targets=None,
            _semantic=None,
        ):
            del cache_modifier
            if multicast_targets is None:
                multicast_targets = []
            eviction_policy = triton_language._unwrap_if_constexpr(
                eviction_policy
            )
            if eviction_policy not in ("", "evict_first", "evict_last"):
                raise ValueError(f"invalid eviction policy: {eviction_policy}")
            if not all(
                hasattr(desc, name)
                for name in ("handle", "block_shape", "dtype")
            ):
                raise TypeError(f"expected a tensor descriptor, got {type(desc)}")
            if len(offsets) != len(desc.block_shape):
                raise ValueError("descriptor offsets must match its rank")
            result_handle = require_nv_mma_shared_layout(
                result, True, _semantic.builder
            )
            offset_handles = _semantic._convert_to_ir_values(
                offsets, require_i64=False
            )
            pred_handle = (
                _semantic.builder.get_int1(True)
                if pred is None
                else pred.handle
            )
            _semantic.builder.create_async_tma_copy_global_to_local(
                desc.handle,
                offset_handles,
                barrier.handle,
                result_handle,
                pred_handle,
                len(multicast_targets) > 0,
                None,
            )

        tlx.async_descriptor_load = _async_descriptor_load
        mem_ops.async_descriptor_load = _async_descriptor_load
        tlx._layout_descriptor_compatible = True

    # Stock Triton 3.7 injects `_semantic` when it evaluates a JIT context
    # manager. triton-utlx 3.7.1's outer task-group constructor predates that
    # keyword, although its inner async_task constructor already accepts it.
    if not getattr(tlx.async_tasks, "_stock_triton_37_compatible", False):

        def _async_tasks_init(self, _semantic=None, _builder=None):
            del _semantic, _builder

        tlx.async_tasks.__init__ = _async_tasks_init
        tlx.async_tasks._stock_triton_37_compatible = True

    if not getattr(tlx.async_task, "_stock_triton_37_compatible", False):
        original_async_task_init = tlx.async_task.__init__

        def _async_task_init(
            self, *args, _builder=None, _semantic=None, **kwargs
        ):
            if _builder is None and _semantic is not None:
                _builder = _semantic.builder
            original_async_task_init(
                self, *args, _builder=_builder, **kwargs
            )

        tlx.async_task.__init__ = _async_task_init
        tlx.async_task._stock_triton_37_compatible = True

    # triton-utlx ships compiler handlers for async_tasks/async_task, but stock
    # Triton 3.7 lacks the WITH_DISPATCH hook that invokes them. Add the narrow
    # hook without changing handling for ordinary Python context managers.
    import ast
    import triton.compiler.code_generator as code_generator
    from utlx_plugin.compiler.dispatch import TLX_WITH_DISPATCH

    if not getattr(
        code_generator.CodeGenerator,
        "_utlx_with_dispatch_compatible",
        False,
    ):
        original_visit_with = code_generator.CodeGenerator.visit_With

        def _visit_with(self, node):
            if len(node.items) == 1:
                context = node.items[0].context_expr
                if isinstance(context, ast.Call):
                    context_fn = self.visit(context.func)
                    handler = TLX_WITH_DISPATCH.get(context_fn)
                    if handler is not None:
                        # This Triton revision predates CodeGenerator.used_vars.
                        # Track name lookups only while TLX builds its isolated
                        # partitions so the handler can derive explicit captures.
                        had_used_vars = hasattr(self, "used_vars")
                        previous_used_vars = getattr(self, "used_vars", None)
                        previous_dereference = self.dereference_name
                        self.used_vars = {
                            child.id
                            for child in ast.walk(node)
                            if isinstance(child, ast.Name)
                            and isinstance(child.ctx, ast.Load)
                        }

                        def _track_name(name):
                            self.used_vars.add(name)
                            return previous_dereference(name)

                        self.dereference_name = _track_name
                        try:
                            return handler(self, node)
                        finally:
                            self.dereference_name = previous_dereference
                            if had_used_vars:
                                self.used_vars = previous_used_vars
                            else:
                                del self.used_vars
            return original_visit_with(self, node)

        code_generator.CodeGenerator.visit_With = _visit_with
        code_generator.CodeGenerator._utlx_with_dispatch_compatible = True

    return tlx
