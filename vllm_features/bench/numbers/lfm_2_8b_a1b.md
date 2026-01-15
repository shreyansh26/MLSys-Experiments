# Prefill

============ Serving Benchmark Result ============
Successful requests:                     512       
Failed requests:                         0         
Benchmark duration (s):                  50.87     
Total input tokens:                      4193792   
Total generated tokens:                  512       
Request throughput (req/s):              10.06     
Output token throughput (tok/s):         10.06     
Peak output token throughput (tok/s):    11.00     
Peak concurrent requests:                512.00    
Total token throughput (tok/s):          82451.94  
---------------Time to First Token----------------
Mean TTFT (ms):                          25793.19  
Median TTFT (ms):                        25786.62  
P99 TTFT (ms):                           50212.61  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================


# Decode

============ Serving Benchmark Result ============
Successful requests:                     512       
Failed requests:                         0         
Benchmark duration (s):                  30.74     
Total input tokens:                      512       
Total generated tokens:                  65536     
Request throughput (req/s):              16.65     
Output token throughput (tok/s):         2131.64   
Peak output token throughput (tok/s):    2224.00   
Peak concurrent requests:                512.00    
Total token throughput (tok/s):          2148.29   
---------------Time to First Token----------------
Mean TTFT (ms):                          15266.31  
Median TTFT (ms):                        14933.56  
P99 TTFT (ms):                           29780.21  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.28      
Median TPOT (ms):                        7.29      
P99 TPOT (ms):                           7.45      
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.29      
Median ITL (ms):                         7.26      
P99 ITL (ms):                            9.85      
==================================================