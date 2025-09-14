import torch
assert torch.cuda.is_available()

torch.manual_seed(0)
C = torch.randn(1000, device='cuda')*2*3 # sanity check for expected A * 2 * 3 value
torch.cuda.synchronize()
torch.manual_seed(0)

stream0 = torch.cuda.Stream()
stream1 = torch.cuda.Stream()

# Stream 0
with torch.cuda.stream(stream0):
    A = torch.randn(1000, device='cuda')
    A *= 2
    Ev = torch.cuda.Event()
    Ev.record(stream0)
    D = torch.randn(1000, device='cuda')

# Stream 1
with torch.cuda.stream(stream1):
    stream1.wait_event(Ev)
    # attempted extra delay to trigger, didn't work:
    # for i in range(10): torch.randn(1024,1024, device='cuda') @ torch.randn(1024,1024, device='cuda')
    B = A * 3

# Prematurely delete A on the CPU
del A
torch.cuda.synchronize()

# this assert never fails.
assert torch.equal(B,C), "memory error triggered!"