'''
Credits - https://github.com/ELS-RD/kernl
'''

import torch
from src.patch_linear_layer import patch_linear_layer
import torch._dynamo as torchdynamo
from typing import List
from torch._subclasses import FakeTensor
from typing import Callable, Iterable, Optional, Union
import math
from torch._inductor.compile_fx import cudagraphify_impl
from torch._dynamo import utils as dynamo_utils

static_inputs_pool = []

class CudaGraphPool:
    """
    Memory pool for CUDA graphs.
    """

    def __init__(self, size, device="cuda"):
        """
        :param size: size of the pool in bytes.
        """
        assert (size > 0) and (size % 8 == 0), f"Size must be positive and multiple of 8, got {size}"
        self.pool: torch.Tensor = torch.empty(size, dtype=torch.int8, device=device)
        self.size = len(self.pool.untyped_storage())
        self.offset = 0

    def copy_to_pool(self, t: torch.Tensor) -> torch.Tensor:
        """
        Copy the tensor t in the pool and return a tensor that is a view of the pool.
        :param t: tensor to copy in the pool
        :return: tensor copy (that is a view of the pool)
        """

        assert t.device == self.pool.device
        assert self.can_store(t)
        # 64 bits alignment
        tensor_aligned_size = get_aligned_size(t)
        new_offset = self.offset + tensor_aligned_size
        # removes 0s from stride
        stride_fixed = tuple(i if i > 0 else 1 for i in t.stride())
        # offset is expressed in t.dtype number of elements
        new_t = torch.as_strided(
            self.pool.view(t.dtype), size=t.size(), stride=stride_fixed, storage_offset=self.offset // t.element_size()
        )
        new_t.copy_(t)
        self.offset = new_offset
        return new_t

    def can_store(self, t: torch.Tensor) -> bool:
        """
        Check if the tensor t can be stored in the pool.
        :param t: tensor to check
        :return: True if the tensor can be stored in the pool
        """
        return (self.pool.device == t.device) and (self.size - self.offset >= get_aligned_size(t))

    def reset(self):
        """
        Reset the pool offset to 0.
        """
        self.offset = 0

def get_aligned_size(t: torch.Tensor, alignment=8) -> int:
    """
    Get the aligned size of the tensor t.
    :param t: tensor to get the aligned size of
    :param alignment: alignment size
    :return: aligned size
    """
    storage_len = len(t.untyped_storage())
    alined_storage_len = (storage_len + alignment - 1) // alignment * alignment
    return alined_storage_len

def get_pool_size(inputs: list[torch.Tensor], existing_pools: list[CudaGraphPool]) -> int:
    """
    Get the size of the pool to use for the CUDA graphs:
    - pool size should be at least as big as the largest existing pool size
    - if pool size < 1Gb, increase its size up to next power of 2 to avoid having many unusuable small pools

    :param inputs: list of inputs to be copied in the pool
    :param existing_pools: list of existing pools
    :return: size of the pool in bytes
    """
    size = sum([get_aligned_size(p) for p in inputs])
    size = max(size, *([p.size for p in existing_pools] + [0]))

    if size < 1024 * 1024 * 1024:
        size = 2 ** math.ceil(math.log2(size))
    return size

def argsort(iterable: Iterable, key: Callable) -> list[int]:
    """
    Sort the list of tensors following provided lambda function.
    :param iterable: iterable object to sort
    :param key: lambda function to sort the iterable object
    :return: indices to sort the iterable object
    """
    return [idx for idx, _ in sorted(enumerate(iterable), key=lambda x: key(x[1]))]

def prepare_inputs(inputs: list[torch.Tensor], pools: list[CudaGraphPool]) -> list[torch.Tensor]:
    """
    Copy the inputs in the CUDA graphs memory pool and return tensor copies.
    Follows a greedy bin packing algorithm (first-fit decreasing) to minimize the number of pools:
    - sort the items in decreasing order of size ;
    - insert them one by one into the first bin that has room for it.

    :param inputs: list of tensors to copy in the pool
    :param pools: list of available pools
    :return: copy of input tensors having their underlying storage in the memory pool
    """
    # reset pool offsets
    for p in pools:
        p.reset()

    pools.sort(key=lambda x: x.size, reverse=False)
    inputs_size_order = argsort(inputs, key=lambda x: x.untyped_storage().size())

    input_copies: list[Optional[torch.Tensor]] = [None] * len(inputs)
    new_pool_index = list()
    for idx in inputs_size_order:
        t = inputs[idx]
        new_pool = True
        for pool in pools:
            if pool.can_store(t):
                new_pool = False
                new_t = pool.copy_to_pool(t)
                input_copies[idx] = new_t
                break
        if new_pool:
            new_pool_index.append(idx)

    if len(new_pool_index) > 0:
        pool_size = get_pool_size(inputs=[inputs[i] for i in new_pool_index], existing_pools=pools)
        new_pool = CudaGraphPool(pool_size, device=inputs[0].device)
        pools.append(new_pool)

        for idx in new_pool_index:
            t = inputs[idx]
            assert new_pool.can_store(t)
            new_t = new_pool.copy_to_pool(t)
            input_copies[idx] = new_t

    return input_copies

def cuda_graphs_wrapper(model: Callable, inputs: Union[list[torch.Tensor], tuple[torch.Tensor]]) -> Callable:
    """
    Wrapper to run the model with cuda graphs.
    @param model: model to save as a CUDA graph
    @param inputs: inputs to the model
    @return: an inference function that runs the model with cuda graphs
    """

    assert isinstance(inputs, (list, tuple))
    # if using fake tensors, defer CUDA graphs until we get real inputs at runtime
    if not any(isinstance(inp, FakeTensor) for inp in inputs):
        inputs = prepare_inputs(inputs=inputs, pools=static_inputs_pool)
        model(*inputs)  # additional warmup needed when input is mutated by some kernel
        f = cudagraphify_impl(
            model=lambda args: model(*args), inputs=inputs, static_input_idxs=tuple(range(len(inputs)))
        )
        return lambda *args: f(prepare_inputs(inputs=args, pools=static_inputs_pool))

    compiled_fn = None

    def run(*new_inputs):
        new_inputs = prepare_inputs(inputs=list(new_inputs), pools=static_inputs_pool)
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                model(*new_inputs)  # additional warmup needed when input is mutated by some kernel
                f = cudagraphify_impl(
                    model=lambda args: model(*args), inputs=new_inputs, static_input_idxs=tuple(range(len(inputs)))
                )

                def compiled_fn(args):
                    return f(list(args))

        return compiled_fn(new_inputs)

    return run

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    patch_linear_layer(gm)
    # print(gm.code)
    return cuda_graphs_wrapper(gm, example_inputs)

def patch_model(model):
    model.orig_forward = model.forward

    static_inputs_pool.clear()

    @torchdynamo.optimize(my_compiler)
    def run(*args, **kwargs):
        return model.orig_forward(*args, **kwargs)
    
    model.forward = run