#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

template<int BM, int BN, int BK, int QSIZE>
struct SMemQueue {
    alignas(128) bf16 A[QSIZE*BM*BK];
    alignas(128) bf16 B[QSIZE*BK*BN];
};

template<int BM, int BN, int BK>
struct SMem {
    alignas(128) bf16 A[BM*BK];
    alignas(128) bf16 B[BK*BN];
};

template <int BM, int BN, int BK, int QSIZE>
struct SMemQueue2 {
    alignas(128) bf16 A[BM*BK*QSIZE];
    alignas(128) bf16 B[BK*BN*QSIZE];
    alignas(128) bf16 C[BN*BM];
};