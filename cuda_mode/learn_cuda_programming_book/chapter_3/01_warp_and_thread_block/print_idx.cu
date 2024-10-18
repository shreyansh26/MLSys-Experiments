#include <iostream>

__global__ void print_idx() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = idx / warpSize;

    // a % b and a & (b-1) are equivalent for b = 2^n
    // int lane_idx = idx % warpSize;
    int lane_idx = idx & (warpSize - 1);

    if((lane_idx & (warpSize/2 - 1)) == 0) {
        printf("%5d\t%5d\t%2d\t%2d\n", idx, blockIdx.x, warp_idx, lane_idx);
    }
}

int main() {
    int gridSize = 4, blockSize = 128;
    std::cout<<"thread, block, warp, lane"<<std::endl;
    print_idx<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    return 0;
}