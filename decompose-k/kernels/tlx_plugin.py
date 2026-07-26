"""Load the TLX compiler plugin with stock Triton 3.7 compatibility."""

from __future__ import annotations

import os
import sysconfig
from types import ModuleType


def configure_tlx_plugin() -> None:
    """Point Triton at the TLX shared library before Triton is imported."""

    if "TRITON_PLUGIN_PATHS" in os.environ:
        return
    purelib = sysconfig.get_paths()["purelib"]
    plugin_path = os.path.join(purelib, "utlx_plugin", "libutlx.so")
    if not os.path.isfile(plugin_path):
        raise RuntimeError(
            f"TLX plugin not found at {plugin_path}; install triton-utlx first"
        )
    os.environ["TRITON_PLUGIN_PATHS"] = plugin_path


def _install_stock_triton_frontend_compat() -> None:
    """Supply frontend helpers expected by the TLX 3.7 plugin DSL."""

    import triton.language as tl
    import triton.language.core as tl_core
    import triton.language.semantic as triton_semantic
    from triton import knobs

    if not hasattr(tl, "_unwrap_if_constexpr"):
        tl._unwrap_if_constexpr = tl_core._unwrap_if_constexpr

    if not hasattr(triton_semantic.TritonSemantic, "_prepare_legacy_load"):

        def _prepare_legacy_load(
            self,
            ptr,
            mask,
            other,
            boundary_check,
            padding,
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
                    ptr,
                    tl.pointer_type(element_ty, ptr_ty.address_space),
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

            m = lhs.type.shape[-2]
            n = rhs.type.shape[-1]
            batch = lhs.type.shape[0] if len(lhs.shape) == 3 else None
            result_ty = tl.block_type(
                result_scalar_ty,
                [batch, m, n] if batch else [m, n],
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
    """Configure the compiler plugin, install compatibility, and return TLX."""

    configure_tlx_plugin()
    _install_stock_triton_frontend_compat()
    import utlx_plugin as tlx

    return tlx
