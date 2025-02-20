#pragma once
#include <cuda_fp16.h>

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
}

// loads an MMA tile directly from global memory
// this is innefficient, access pattern results in bad coalescing
__device__ __forceinline__ void ldmatrix_m16n8_gmem(half* src, half (&reg)[4], unsigned int src_stride_bytes) {
    const unsigned int laneIdx = threadIdx.x % 32;
    uint32_t (&reg_) [2] = reinterpret_cast<uint32_t(&)[2]>(reg);
    uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src);
    src_stride_bytes /= sizeof(uint32_t);
    unsigned int fragment_row = laneIdx / 4;
    const unsigned int fragment_col = laneIdx % 4;
    
    // 4 adjacent threads storing 4 bytes each == 16 byte transactions
    reg_[0] = src_ptr[fragment_row * src_stride_bytes + fragment_col];
    fragment_row += 8;
    reg_[1] = src_ptr[fragment_row * src_stride_bytes + fragment_col];
}

__device__ __forceinline__ void stmatrix_m16n8(half* dst, half (&reg)[4], unsigned int dst_stride_bytes) {
    const unsigned int laneIdx = threadIdx.x % 32;
    uint32_t (&reg_) [2] = reinterpret_cast<uint32_t(&)[2]>(reg);
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst);
    dst_stride_bytes /= sizeof(uint32_t);
    unsigned int fragment_row = laneIdx / 4;
    const unsigned int fragment_col = laneIdx % 4;
    
    // 4 adjacent threads storing 4 bytes each == 16 byte transactions
    dst_ptr[fragment_row * dst_stride_bytes + fragment_col] = reg_[0];
    fragment_row += 8;
    dst_ptr[fragment_row * dst_stride_bytes + fragment_col] = reg_[1];
}

////////////////////////////////////////////////////////////
// Unrolled Loads
////////////////////////////////////////////////////////////


template <unsigned int mma_tiles_per_warp_m, unsigned int mma_tiles_per_warp_k, unsigned int smem_stride>
__device__ __forceinline__ void ldmatrix_a(half* src, half (&reg)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4]) {
    static_assert(mma_tiles_per_warp_m == 8, "mma_tiles_per_warp_m must be 4");
    static_assert(mma_tiles_per_warp_k == 4, "mma_tiles_per_warp_k must be 4");

    uint32_t (&reg_) [mma_tiles_per_warp_m][mma_tiles_per_warp_k][2] = reinterpret_cast<uint32_t(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2]>(reg);
    unsigned int logical_offset = (threadIdx.x % 32) * smem_stride;
    unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b10000000) >> 4);
    swizzled_offset = swizzled_offset ^ ((swizzled_offset & 0b1100000) >> 2);
    uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
    constexpr unsigned int smem_stride_ = smem_stride * sizeof(half); // convert stride to bytes
    
    // 0
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][0][0]), "=r"(reg_[0][0][1]), "=r"(reg_[1][0][0]), "=r"(reg_[1][0][1])
        : "r"(src_addr)
    );

    // 0
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][0][0]), "=r"(reg_[2][0][1]), "=r"(reg_[3][0][0]), "=r"(reg_[3][0][1])
        : "r"(src_addr + 32 * smem_stride_)
    );

    // 0
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][0][0]), "=r"(reg_[4][0][1]), "=r"(reg_[5][0][0]), "=r"(reg_[5][0][1])
        : "r"(src_addr + 64 * smem_stride_)
    );

    // 0
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[6][0][0]), "=r"(reg_[6][0][1]), "=r"(reg_[7][0][0]), "=r"(reg_[7][0][1])
        : "r"(src_addr + 96 * smem_stride_)
    );

    src_addr ^= 0b10000;
    
    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][1][0]), "=r"(reg_[0][1][1]), "=r"(reg_[1][1][0]), "=r"(reg_[1][1][1])
        : "r"(src_addr)
    );

    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][1][0]), "=r"(reg_[2][1][1]), "=r"(reg_[3][1][0]), "=r"(reg_[3][1][1])
        : "r"(src_addr + 32 * smem_stride_)
    );

    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][1][0]), "=r"(reg_[4][1][1]), "=r"(reg_[5][1][0]), "=r"(reg_[5][1][1])
        : "r"(src_addr + 64 * smem_stride_)
    );

    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[6][1][0]), "=r"(reg_[6][1][1]), "=r"(reg_[7][1][0]), "=r"(reg_[7][1][1])
        : "r"(src_addr + 96 * smem_stride_)
    );
    
    src_addr ^= 0b110000;

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][2][0]), "=r"(reg_[0][2][1]), "=r"(reg_[1][2][0]), "=r"(reg_[1][2][1])
        : "r"(src_addr)
    );

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][2][0]), "=r"(reg_[2][2][1]), "=r"(reg_[3][2][0]), "=r"(reg_[3][2][1])
        : "r"(src_addr + 32 * smem_stride_)
    );

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][2][0]), "=r"(reg_[4][2][1]), "=r"(reg_[5][2][0]), "=r"(reg_[5][2][1])
        : "r"(src_addr + 64 * smem_stride_)
    );

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[6][2][0]), "=r"(reg_[6][2][1]), "=r"(reg_[7][2][0]), "=r"(reg_[7][2][1])
        : "r"(src_addr + 96 * smem_stride_)
    );
    src_addr ^= 0b10000;

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][3][0]), "=r"(reg_[0][3][1]), "=r"(reg_[1][3][0]), "=r"(reg_[1][3][1])
        : "r"(src_addr)
    );

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][3][0]), "=r"(reg_[2][3][1]), "=r"(reg_[3][3][0]), "=r"(reg_[3][3][1])
        : "r"(src_addr + 32 * smem_stride_)
    );

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][3][0]), "=r"(reg_[4][3][1]), "=r"(reg_[5][3][0]), "=r"(reg_[5][3][1])
        : "r"(src_addr + 64 * smem_stride_)
    );

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[6][3][0]), "=r"(reg_[6][3][1]), "=r"(reg_[7][3][0]), "=r"(reg_[7][3][1])
        : "r"(src_addr + 96 * smem_stride_)
    );

}

template <unsigned int mma_tiles_per_warp_k, unsigned int mma_tiles_per_warp_n, unsigned int smem_stride>
__device__ __forceinline__ void ldmatrix_b(half* src, half (&reg)[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2]) {
    static_assert(mma_tiles_per_warp_k == 4, "mma_tiles_per_warp_k must be 4");
    static_assert(mma_tiles_per_warp_n == 8, "mma_tiles_per_warp_n must be 8");

    uint32_t (&reg_) [4][8] = reinterpret_cast<uint32_t(&)[4][8]>(reg);
    const unsigned int logical_offset = ((threadIdx.x % 8) * smem_stride) +  (((threadIdx.x % 32) / 8) * 8);
    unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b11100000000) >> 5);
    uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
    constexpr unsigned int smem_stride_ = smem_stride * sizeof(half); // convert stride to bytes

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][0]), "=r"(reg_[0][1]), "=r"(reg_[0][2]), "=r"(reg_[0][3])
        : "r"(src_addr)
    );

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][4]), "=r"(reg_[0][5]), "=r"(reg_[0][6]), "=r"(reg_[0][7])
        : "r"(src_addr ^ 0b1000000)
    );

    src_addr += 8 * smem_stride_;

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[1][0]), "=r"(reg_[1][1]), "=r"(reg_[1][2]), "=r"(reg_[1][3])
        : "r"(src_addr)
    );

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[1][4]), "=r"(reg_[1][5]), "=r"(reg_[1][6]), "=r"(reg_[1][7])
        : "r"(src_addr ^ 0b1000000)
    );

    src_addr += 8 * smem_stride_;

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][0]), "=r"(reg_[2][1]), "=r"(reg_[2][2]), "=r"(reg_[2][3])
        : "r"(src_addr)
    );

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][4]), "=r"(reg_[2][5]), "=r"(reg_[2][6]), "=r"(reg_[2][7])
        : "r"(src_addr ^ 0b1000000)
    );

    src_addr += 8 * smem_stride_;

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[3][0]), "=r"(reg_[3][1]), "=r"(reg_[3][2]), "=r"(reg_[3][3])
        : "r"(src_addr)
    );

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[3][4]), "=r"(reg_[3][5]), "=r"(reg_[3][6]), "=r"(reg_[3][7])
        : "r"(src_addr ^ 0b1000000)
    );

}