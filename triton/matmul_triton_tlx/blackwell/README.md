# Blackwell-native kernels

These kernels are separate from the Hopper implementations because their
accumulation and scheduling model is materially different on SM100:

- `tlx_matmul.py` uses dedicated TMA, TCGen5 MMA, and epilogue partitions.
  Operands are staged through SMEM, FP32 accumulators live in double-buffered
  TMEM, and `tcgen05.commit` barriers connect the three partitions.
- `gluon_matmul.py` implements the same one-CTA pipeline with Gluon's native
  `TensorMemoryLayout`, `tcgen05_mma`, and `gl.warp_specialize` APIs. Its
  epilogue is split four ways so the four-stage operand pipeline and two TMEM
  accumulator tiles fit without materializing the full output tile in SMEM.
- `gluon_matmul_2cta.py` pairs two CTAs along M. TMA multicast shares operand
  tiles across the cluster, collaborative TCGen5 produces both output halves,
  and a fourth specialized partition drives Cluster Launch Control. Six input
  stages and four rotating epilogue buffers hide the cluster latency.

The one-CTA kernels use grouped, SM-stride persistent scheduling and fixed
`128 x 256 x 64` tiles. Two-CTA Gluon computes a logical `256 x 256 x 64`
cluster tile and uses CLC work stealing. Its public wrapper selects two CTAs
only when `M`, `N`, and `K` are all at least 4096; `matmul_one_cta` and
`matmul_two_cta` remain available for direct benchmarking. The top-level TLX
and Gluon files remain the readable Hopper/WGMMA versions.
