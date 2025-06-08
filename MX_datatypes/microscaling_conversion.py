"""
mx_converter.py  –  minimal “research-grade” implementation of Algorithm 1
=========================================================================

* Works on any FP32 tensor (CPU or CUDA) and returns
  (scale_tensor, packed_tensor) for the chosen MX format.
* Block size is fixed at 32, because all MX variants in the
  OCP spec and NVIDIA Blackwell docs use 32-value blocks :contentReference[oaicite:0]{index=0}.
* The scale is stored as an 8-bit **E8M0** power-of-two,
  i.e.   stored_byte = shared_exp + 127   so that
  `float_scale = 2**(stored_byte-127)`.
  (E8M0 is unsigned, mantissa = 0, so only the exponent matters.)
* For the element quantiser we rely on the new PyTorch 2.3
  FP8 dtypes (`torch.float8_e4m3fn`, `torch.float8_e5m2`) when
  they are available.  Otherwise we fall back to a very
  simple “round-to-nearest-power-of-two” emulation so the
  script remains runnable on any machine.

Author: 2025-06-08
"""
import math
from typing import Tuple

import torch

# --------------------------------------------------------------------------
# 1.  MX–format bookkeeping
# --------------------------------------------------------------------------
_MX_SPECS = {
    # name          emax_elem,      torch dtype (if present)   bits
    "MXFP8_E4M3": (8,  torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None, 8),
    "MXFP8_E5M2": (15, torch.float8_e5m2   if hasattr(torch, "float8_e5m2")   else None, 8),
    "MXFP6_E2M3": (1,  None, 6),             # 6-bit float (no native dtype yet)
    "MXINT8":     (7,  torch.int8, 8),        # signed int8 in [-128, 127]
}

_BLOCK_SIZE = 32                 # K in the paper / OCP spec
_E8M0_BIAS  = 127                # power-of-two bias for the scale byte


# --------------------------------------------------------------------------
# 2.  Helper: quantise a *single* FP32 value to a target element format
# --------------------------------------------------------------------------
def _quantise_elem(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Quantise x (float32) element-wise to the given MX element format."""
    emax_elem, torch_dtype, bits = _MX_SPECS[fmt]
    if torch_dtype is not None and torch_dtype.is_floating_point:
        # Native FP8 casting path (fast, needs PyTorch≥2.3 and sm_90 for CUDA)
        return x.to(dtype=torch_dtype)
    elif fmt == "MXINT8":
        # Symmetric int8: round-to-nearest and clamp to representable range
        return torch.clamp(torch.round(x), -128, 127).to(torch.int8)
    else:
        # Crude floating-point emulation for 6-bit or FP8 when native dtype missing.
        # We quantise the mantissa to 2**(−mant_bits) grid *inside* each binade.
        mant_bits = bits - 1 - (3 if fmt.endswith("E2M3") else 4 if fmt.endswith("E4M3") else 5)
        step = 2.0 ** (-mant_bits)
        quantised = torch.round(x / step) * step      # round mantissa
        # Clamp to numeric range ±2**emax_elem  (approximate)
        max_val = (2 ** emax_elem) * (1 - step)
        return torch.clamp(quantised, -max_val, max_val)
    

# --------------------------------------------------------------------------
# 3.  Algorithm 1 – vector / tensor version
# --------------------------------------------------------------------------
def float32_to_mx(
    tensor_fp32: torch.Tensor,
    fmt: str = "MXFP8_E4M3",
    block_axis: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a float32 tensor to MX format.

    Returns
    -------
    scale_e8m0  : uint8 tensor with shape == tensor.shape with `block_axis`
                  collapsed to blocks of 32.   1 byte per block.
    payload     : tensor of the element dtype specified by the MX format,
                  same shape as `tensor_fp32`.
    """
    assert fmt in _MX_SPECS, f"Unknown MX format '{fmt}'."

    # Move the chosen axis to the end so `.view` works block-wise
    t = tensor_fp32.transpose(block_axis, -1).contiguous()
    orig_shape = t.shape
    last_dim   = orig_shape[-1]
    pad = (-last_dim) % _BLOCK_SIZE           # pad so it's divisible by 32
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))

    t2 = t.view(*t.shape[:-1], -1, _BLOCK_SIZE)   # (..., n_blocks, 32)

    # Step (1) – compute shared exponent per block
    max_abs     = t2.abs().amax(dim=-1)           # (..., n_blocks)
    emax_elem   = _MX_SPECS[fmt][0]
    shared_exp  = torch.floor(torch.log2(max_abs + 1e-30)).to(torch.int32) - emax_elem
    shared_exp  = shared_exp.clamp(min=-(2**7), max=(2**7)-1)   # keep in signed int8 range

    # Step (2) – scale to power-of-two and quantise
    scale = torch.pow(2.0, shared_exp).unsqueeze(-1)            # broadcast over 32
    payload = _quantise_elem(t2 / scale, fmt)

    # Step (3) – pack scale as unsigned E8M0 byte
    print(scale)
    scale_e8m0 = (shared_exp + _E8M0_BIAS).to(torch.uint8)

    # Undo padding & axis move
    payload    = payload.view(*orig_shape).transpose(block_axis, -1)
    scale_e8m0 = scale_e8m0.view(*orig_shape[:-1], -1).transpose(block_axis, -1)

    return scale_e8m0, payload

if __name__ == "__main__":  
    x = torch.randn(4, 128, device="cuda") * 123.4   # FP32 activations
    scale, data = float32_to_mx(x, fmt="MXFP8_E4M3")  # block-scaled FP8

    print(scale.shape)   # -> torch.Size([4, 4])   (128 / 32 = 4 blocks per row)
    print(scale.dtype)
    print(data.shape)
    print(data.dtype)    # -> torch.float8_e4m3fn  (if available) or torch.int8 …

    # print(x)
    # print(data)
    # print(scale)
