#include <cstdio>
#include <helper_timer.h>

using namespace std;

/*
cudaEventSynchronize(stop):
* Waits only for operations before the specific stop event to complete
* More efficient as it only synchronizes operations up to that event
* In your code, it waits for the vecAdd_kernel to finish since stop was recorded right after the kernel launch

cudaDeviceSynchronize():
* Waits for ALL operations on the GPU to complete
* Less efficient as it's a global synchronization
* Would synchronize even operations that happened after your kernel (if any)

*/

__global__ void vecAdd_kernel(float *c, const float* a, const float* b);
void init_buffer(float *data, const int size);

int main(int argc, char* argv[]) {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);

    // allocate host memories
    cudaMallocHost((void**)&h_a, bufsize);
    cudaMallocHost((void**)&h_b, bufsize);
    cudaMallocHost((void**)&h_c, bufsize);

    // initialize host values
    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice);

    // initialize the host timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    cudaEvent_t start, stop;
    // create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to measure the execution time
    sdkStartTimer(&timer);
    cudaEventRecord(start);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock >>>(d_c, d_a, d_b);

    // record the event right after the kernel execution finished
    cudaEventRecord(stop);

    // Wait for kernel to finish
    cudaEventSynchronize(stop);  // Only waits for operations up to 'stop' event
    // cudaDeviceSynchronize();  // Would wait for ALL GPU operations

    sdkStopTimer(&timer);
    
    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost);

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // print estimated kernel execution time
    float elapsed_time_msed = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed, start, stop);
    printf("CUDA event estimated = %.3f ms \n", elapsed_time_msed);

    // Compute and print the performance
    elapsed_time_msed = sdkGetTimerValue(&timer);
    printf("Host measured time = %.3f ms\n", elapsed_time_msed);

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    // delete timer
    sdkDeleteTimer(&timer);

    // terminate CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

__global__ void vecAdd_kernel(float *c, const float* a, const float* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

void init_buffer(float *data, const int size) {
    for(int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}