// Based on - https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
int main() {
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
    cudaMalloc((int**)&d_a, bytes);

    memset(h_a, 0, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    return 0;
}


/*
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

Collecting data...
Generating '/tmp/nsys-report-06a9.qdstrm'
[1/8] [========================100%] report2.nsys-rep
[2/8] [========================100%] report2.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /teamspace/studios/this_studio/report2.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)       Name
 --------  ---------------  ---------  ----------  ---------  --------  ---------  -----------  --------------
     69.8        480897689          8  60112211.1  2457525.5     20179  319278948  116425722.9  sem_wait
     19.6        134870020         12  11239168.3  2117280.5      3881   48255299   17733373.0  poll
     10.0         69186505        514    134604.1    11574.0      1039   27284720    1282314.2  ioctl
      0.3          2114276         31     68202.5     7143.0      4391    1597102     284282.4  mmap64
      0.1           797892         49     16283.5    16552.0      5283      25130       4683.6  open64
      0.1           591980         10     59198.0    28795.0     11096     371774     110265.1  sem_timedwait
      0.0           214064          4     53516.0    55505.5     41121      61932       8839.4  pthread_create
      0.0           195228         43      4540.2     3228.0      1844      13198       3212.9  fopen
      0.0           135345         15      9023.0     3483.0      1451      62681      15510.5  mmap
      0.0            63048          1     63048.0    63048.0     63048      63048          0.0  fgets
      0.0            55430         10      5543.0     5366.0      3290      11186       2157.1  write
      0.0            43053         28      1537.6     1375.5      1044       2868        492.4  fclose
      0.0            35689          7      5098.4     4561.0      1910       7487       2087.6  open
      0.0            24669          3      8223.0     6745.0      6732      11192       2571.2  fread
      0.0            22225          5      4445.0     3997.0      2644       5968       1380.1  munmap
      0.0            21419         10      2141.9     1857.0      1097       4547       1071.0  read
      0.0            19989         17      1175.8     1107.0      1031       1578        147.1  fcntl
      0.0            14213          2      7106.5     7106.5      5533       8680       2225.3  socket
      0.0             9208          1      9208.0     9208.0      9208       9208          0.0  connect
      0.0             8518          1      8518.0     8518.0      8518       8518          0.0  pipe2
      0.0             2067          1      2067.0     2067.0      2067       2067          0.0  bind
      0.0             1191          1      1191.0     1191.0      1191       1191          0.0  listen

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)           Name
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------
     98.7        148778330          1  148778330.0  148778330.0  148778330  148778330          0.0  cudaMalloc
      1.3          1960451          2     980225.5     980225.5     937199    1023252      60848.7  cudaMemcpy
      0.0             1510          1       1510.0       1510.0       1510       1510          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report
SKIPPED: /teamspace/studios/this_studio/report2.sqlite does not contain CUDA kernel data.
[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)           Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
     54.3           811171      1  811171.0  811171.0    811171    811171          0.0  [CUDA memcpy Host-to-Device]
     45.7           683912      1  683912.0  683912.0    683912    683912          0.0  [CUDA memcpy Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
      4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Device-to-Host]
      4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Host-to-Device]

Generated:
    /teamspace/studios/this_studio/report2.nsys-rep
    /teamspace/studios/this_studio/report2.sqlite
*/

/*
Pinned memory is used as a staging area for transfers from the device to the host. 
We can avoid the cost of the transfer between pageable and pinned host arrays by directly allocating our host arrays in pinned memory.
*/