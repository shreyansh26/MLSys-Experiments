import math

import triton
import triton.language as tl

sqrt2pi = math.sqrt(2.0 / math.pi)
sqrt2 = math.sqrt(2.0)

@triton.jit
def tanh_triton_fwd(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def tanh_triton_bwd(x):
    a = tanh_triton_fwd(x)
    return 1 - a*a

@triton.jit
def sigmoid_triton_fwd(x):
    return tl.sigmoid(x)

@triton.jit
def sigmoid_triton_bwd(x):
    a = sigmoid_triton_fwd(x)
    return a * (1 - a)

@triton.jit
def relu_triton_fwd(x):
    return tl.maximum(0, x)

@triton.jit
def relu_triton_bwd(x):
    return tl.where(x >= 0, 1.0, 0.0)

@triton.jit
def leaky_relu_triton_fwd(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.jit
def leaky_relu_triton_bwd(x):
    return tl.where(x >= 0.0, 1.0, 0.01)

@triton.jit
def gelu_triton_fwd(x):
    return 0.5 * x * (1 + tl.libdevice.erf(x / sqrt2))

@triton.jit
def fast_gelu_triton_fwd(x):
    return 0.5 * x * (1 + tanh_triton_fwd(sqrt2pi * (x + 0.044715 * x * x * x)))

@triton.jit
def fast_gelu_triton_bwd(x):
    tanh_out = tanh_triton_fwd(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)