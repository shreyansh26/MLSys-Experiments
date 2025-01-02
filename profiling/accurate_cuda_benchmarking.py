# From - https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
# This is also similar to what triton's do_bench does
import os
import subprocess

# allocating 40MB to match L2 cache size on A100
x = torch.empty(int(40 * (1024 ** 2)), dtype=torch.int8, device='cuda')

DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")
CLOCK_SPEED = 1350 # Must choose a clock speed that's supported on your device.

def set_clock_speed():
    """
    Set GPU clock speed to a specific value.
    This doesn't guarantee a fixed value due to throttling, but can help reduce variance.
    """
    process = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True)
    stdout, _ = process.communicate()
    process = subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {DEVICE}",      shell=True)
    process = subprocess.run(f"sudo nvidia-smi -lgc {CLOCK_SPEED} -i {DEVICE}", shell=True)

def reset_clock_speed():
    """
    Reset GPU clock speed to default values.
    """
    subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {DEVICE}", shell=True)
    subprocess.run(f"sudo nvidia-smi -rgc -i {DEVICE}", shell=True)

def flush_cache():
    x.zero_()
     
# To make benchmarking reproducible
set_clock_speed()

steps = 10

# Warmup steps
for _ in range(steps):
    run_kernel()

start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

for i in range(steps):
    # Flush cache due to write-back policy of L2 cache in NVIDIA
    flush_cache()
    # Saturate GPUs / Ensure no process queued before kernel call so that kernel and its events are enqueued together
    torch.cuda._sleep(1_000_000)

    start_events[i].record()
    run_kernel()
    end_events[i].record()

torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

reset_clock_speed()
