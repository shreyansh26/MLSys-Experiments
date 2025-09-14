import torch
from torch import Tensor

def softmax(logits: Tensor) -> Tensor:
    a_max = logits.max(axis=-1, keepdim=True)
    a_exp = torch.exp(logits - a_max.values)
    return a_exp / a_exp.sum(axis=-1, keepdim=True)

@torch.compile
def segment_softmax(logits: Tensor, segments: Tensor, max_segments: int) -> Tensor:
    """
    Segment-wise softmax over the first dimension.
    logits:   [N, ...]
    segments: [N] with values in [0, max_segments-1]
    Returns a tensor shaped like logits with softmax computed within each segment.
    """
    # Ensure index type
    segments = segments.to(torch.long)

    # Build an index that broadcasts over trailing dims so scatter/gather can work on dim=0
    expand_dims = (1,) * (logits.ndim - 1)
    # print(logits.ndim, expand_dims)
    idx = segments.view(-1, *expand_dims).expand_as(logits)
    # print(idx)

    # 1) Per-segment max for numerical stability
    seg_shape = (max_segments,) + tuple(logits.shape[1:])
    # print(seg_shape)
    seg_max = torch.full(seg_shape, float("-inf"), dtype=logits.dtype, device=logits.device)
    # print(seg_max)
    seg_max.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)
    # print(seg_max)

    # Shift by segment max and exponentiate
    # print(seg_max.gather(0, idx))
    x = logits - seg_max.gather(0, idx)
    ex = torch.exp(x)

    # 2) Per-segment sum of exp
    seg_sum = torch.zeros_like(seg_max)
    seg_sum.scatter_reduce_(0, idx, ex, reduce="sum", include_self=False)

    # Normalize by gathered segment sums
    return ex / seg_sum.gather(0, idx)


if __name__ == "__main__":
    logits = torch.tensor([-1., 0., 1., 5., 6., -7.])
    segments = torch.tensor([0, 0, 0, 1, 1, 1])
    max_segments = 2
    print(segment_softmax(logits, segments, max_segments))
    print(torch.cat((softmax(logits[:3]), softmax(logits[3:])), axis=0))