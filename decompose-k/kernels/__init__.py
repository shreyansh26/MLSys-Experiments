"""Triton kernel implementations for the Decompose-K experiments."""

__all__ = ["KernelConfig"]


def __getattr__(name: str) -> object:
    if name == "KernelConfig":
        from .decompose_k_triton_kernel import KernelConfig

        return KernelConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
