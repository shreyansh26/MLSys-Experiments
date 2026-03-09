import torch


DELAY = 100000000
try:
    i = -1
    for i in range(100):
        # Create a tensor in pin-memory
        cpu_tensor = torch.ones(1024, 1024, pin_memory=True)
        torch.cuda.synchronize()
        # Send the tensor to CUDA
        cuda_tensor = cpu_tensor.to("cuda", non_blocking=True)
        # torch.cuda.synchronize() # this fixes the issue
        torch.cuda._sleep(DELAY)
        # Corrupt the original tensor
        cpu_tensor.zero_()
        assert (cuda_tensor == 1).all()
    print("No test failed with non_blocking and pinned tensor")
except AssertionError:
    print(f"{i}th test failed with non_blocking and pinned tensor. Skipping remaining tests")


i = -1
for i in range(100):
    # Create a tensor in pageable memory
    cpu_tensor = torch.ones(1024, 1024)
    torch.cuda.synchronize()
    # Send the tensor to CUDA
    cuda_tensor = cpu_tensor.to("cuda", non_blocking=True)
    torch.cuda._sleep(DELAY)
    # Corrupt the original tensor
    cpu_tensor.zero_()
    assert (cuda_tensor == 1).all()
print("No test failed with non_blocking and pageable tensor")