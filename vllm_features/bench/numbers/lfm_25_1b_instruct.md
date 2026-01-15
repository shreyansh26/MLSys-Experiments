# Prefill

============ Serving Benchmark Result ============
Successful requests:                     512       
Failed requests:                         0         
Benchmark duration (s):                  19.92     
Total input tokens:                      4193792   
Total generated tokens:                  512       
Request throughput (req/s):              25.71     
Output token throughput (tok/s):         25.71     
Peak output token throughput (tok/s):    28.00     
Peak concurrent requests:                512.00    
Total token throughput (tok/s):          210603.01 
---------------Time to First Token----------------
Mean TTFT (ms):                          10271.94  
Median TTFT (ms):                        10262.63  
P99 TTFT (ms):                           19561.85  
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
Benchmark duration (s):                  10.03     
Total input tokens:                      512       
Total generated tokens:                  65536     
Request throughput (req/s):              51.03     
Output token throughput (tok/s):         6531.77   
Peak output token throughput (tok/s):    6992.00   
Peak concurrent requests:                512.00    
Total token throughput (tok/s):          6582.80   
---------------Time to First Token----------------
Mean TTFT (ms):                          5093.81   
Median TTFT (ms):                        4982.91   
P99 TTFT (ms):                           9696.32   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2.27      
Median TPOT (ms):                        2.27      
P99 TPOT (ms):                           2.42      
---------------Inter-token Latency----------------
Mean ITL (ms):                           2.29      
Median ITL (ms):                         2.21      
P99 ITL (ms):                            5.62      
==================================================