// Based on - https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
#include <stdio.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if(result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

void profileCopies(float        *h_a, 
                   float        *h_b, 
                   float        *d, 
                   unsigned int  n,
                   char         *desc) {
    printf("\n%s transfers\n", desc);

    unsigned int bytes = n * sizeof(float);

    // events for timing
    cudaEvent_t startEvent, stopEvent; 

    checkCuda(cudaEventCreate(&startEvent) );
    checkCuda(cudaEventCreate(&stopEvent) );

    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    float time;
    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
    printf("Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
    printf("Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    for(int i = 0; i < n; ++i) {
        if(h_a[i] != h_b[i]) {
            printf("*** %s transfers failed ***", desc);
            break;
        }
    }

    // clean up events
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
}

int main() {
    unsigned int nElements = 4 * 1024 * 1024;
    const unsigned int bytes = nElements * sizeof(float);

    // host arrays
    float *h_aPageable, *h_bPageable;   
    float *h_aPinned, *h_bPinned;

    // device array
    float *d_a;

    // allocate and initialize
    h_aPageable = (float*)malloc(bytes);                    // host pageable
    h_bPageable = (float*)malloc(bytes);                    // host pageable
    checkCuda(cudaMallocHost((void**)&h_aPinned, bytes)); // host pinned
    checkCuda(cudaMallocHost((void**)&h_bPinned, bytes)); // host pinned
    checkCuda(cudaMalloc((void**)&d_a, bytes));           // device

    for(int i = 0; i < nElements; ++i) 
        h_aPageable[i] = i;      
    
    memcpy(h_aPinned, h_aPageable, bytes);
    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    // output device info and transfer size
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0) );

    printf("\nDevice: %s\n", prop.name);
    printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

    // perform copies and report bandwidth
    profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

    printf("\n");

    // cleanup
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);

    return 0;
}

/*
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

Collecting data...

Device: Tesla T4
Transfer size (MB): 16

Pageable transfers
Host to Device bandwidth (GB/s): 4.526944
Device to Host bandwidth (GB/s): 4.551822

Pinned transfers
Host to Device bandwidth (GB/s): 6.195207
Device to Host bandwidth (GB/s): 6.569613

Generating '/tmp/nsys-report-38ad.qdstrm'
[1/8] [========================100%] report3.nsys-rep
[2/8] [========================100%] report3.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /teamspace/studios/this_studio/report3.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)       Name     
 --------  ---------------  ---------  ----------  ---------  --------  ---------  -----------  --------------
     73.1        511907520          8  63988440.0  2602040.5     27501  320954997  111610767.8  sem_wait      
     14.3        100011710         12   8334309.2  1612722.5      3095   43196168   14671193.2  poll          
     11.5         80789173        537    150445.4    10867.0      1040   18830325    1197824.1  ioctl         
      0.5          3533186         21    168247.0     3830.0      1569    1757599     505914.8  mmap          
      0.2          1579497         31     50951.5     6875.0      4249    1093707     194233.5  mmap64        
      0.1           786297         49     16046.9    16403.0      4258      25248       4352.7  open64        
      0.1           653993         10     65399.3    33861.5     11095     413912     123113.6  sem_timedwait 
      0.0           205638         43      4782.3     3325.0      1857      14150       3363.3  fopen         
      0.0           186576          4     46644.0    43053.0     37664      62806      11340.2  pthread_create
      0.0            65335          1     65335.0    65335.0     65335      65335          0.0  fgets         
      0.0            59207         11      5382.5     5638.0      1038      10252       2509.9  write         
      0.0            46184         28      1649.4     1459.5      1056       3456        610.3  fclose        
      0.0            44121          9      4902.3     3772.0      2376      10792       2755.5  munmap        
      0.0            38530          7      5504.3     5166.0      2627       8067       2273.8  open          
      0.0            28264          3      9421.3     7972.0      5643      14649       4674.7  fread         
      0.0            24169         10      2416.9     2131.5      1465       4497        930.8  read          
      0.0            15259         12      1271.6     1235.0      1049       1790        219.7  fcntl         
      0.0            15167          2      7583.5     7583.5      6782       8385       1133.5  socket        
      0.0            11012          1     11012.0    11012.0     11012      11012          0.0  connect       
      0.0             8944          1      8944.0     8944.0      8944       8944          0.0  pipe2         
      0.0             2866          1      2866.0     2866.0      2866       2866          0.0  bind          
      0.0             1312          1      1312.0     1312.0      1312       1312          0.0  listen        

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------------
     83.9        109114829          2  54557414.5  54557414.5   7183592  101931237   66996702.3  cudaMallocHost        
      9.6         12476076          4   3119019.0   3123275.5   2550833    3678692     574397.4  cudaMemcpy            
      3.7          4808552          2   2404276.0   2404276.0   2312739    2495813     129452.9  cudaFreeHost          
      1.7          2247342          4    561835.5      2983.0       778    2240598    1119176.2  cudaEventCreate       
      0.6           800292          8    100036.5      4223.0      1427     766779     269431.0  cudaEventRecord       
      0.2           217252          1    217252.0    217252.0    217252     217252          0.0  cudaFree              
      0.1           167698          4     41924.5      3741.5      3077     157138      76809.8  cudaEventSynchronize  
      0.1           126469          1    126469.0    126469.0    126469     126469          0.0  cudaMalloc            
      0.0            19377          4      4844.3      4857.5       630       9032       4747.9  cudaEventDestroy      
      0.0             1145          1      1145.0      1145.0      1145       1145          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report
SKIPPED: /teamspace/studios/this_studio/report3.sqlite does not contain CUDA kernel data.
[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
     50.8          6135336      2  3067668.0  3067668.0   2676482   3458854     553220.5  [CUDA memcpy Host-to-Device]
     49.2          5930287      2  2965143.5  2965143.5   2539111   3391176     602500.9  [CUDA memcpy Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
     33.554      2    16.777    16.777    16.777    16.777        0.000  [CUDA memcpy Device-to-Host]
     33.554      2    16.777    16.777    16.777    16.777        0.000  [CUDA memcpy Host-to-Device]

Generated:
    /teamspace/studios/this_studio/report3.nsys-rep
    /teamspace/studios/this_studio/report3.sqlite
*/