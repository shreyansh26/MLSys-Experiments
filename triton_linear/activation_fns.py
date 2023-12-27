import math

import triton
import triton.language as tl

sqrt2pi = math.sqrt(2.0 / math.pi)
sqrt2 = math.sqrt(2.0)

@triton.jit
def tanh_triton(x):
    return tl.libdevice.tanh(x)

@triton.jit
def sigmoid_triton(x):
    return 1 / (1 + tl.libdevice.exp(-1 * x))

@triton.jit
def relu_triton(x):
    return tl.maximum(0, x)

@triton.jit
def leaky_relu_triton(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.jit
def gelu_triton(x):
    return 0.5 * x * (1 + tl.libdevice.erf(x / sqrt2))

@triton.jit
def fast_gelu_triton(x):
    return 0.5 * x * (1 + tanh_triton(sqrt2pi * (x + 0.044715 * x * x * x)))