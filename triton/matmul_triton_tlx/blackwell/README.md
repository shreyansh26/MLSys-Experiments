# Blackwell-native kernels

These kernels are separate from the Hopper implementations because their
accumulation and scheduling model is materially different on SM100:

- `tlx_matmul.py` uses dedicated TMA, TCGen5 MMA, and epilogue partitions.
  Operands are staged through SMEM, FP32 accumulators live in double-buffered
  TMEM, and `tcgen05.commit` barriers connect the three partitions.
- `gluon_matmul.py` implements the same high-level pipeline with Gluon's native
  `TensorMemoryLayout`, `tcgen05_mma`, and `gl.warp_specialize` APIs. Its
  epilogue is split four ways so the four-stage operand pipeline and two TMEM
  accumulator tiles fit without materializing the full output tile in SMEM.

Both kernels use a grouped, SM-stride persistent scheduler and fixed
`128 x 256 x 64` tiles. `benchmark.py` selects these implementations
automatically on compute capability 10.0 or newer. The top-level TLX and Gluon
files remain the readable Hopper/WGMMA versions.
