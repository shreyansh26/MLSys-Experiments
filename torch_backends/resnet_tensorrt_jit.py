import torch_tensorrt
import torch

# Load a pre-trained ResNet50 model
x = torch.randn(1, 3, 224, 224, device='cuda').half()
model = torch.hub.load(
    'pytorch/vision:v0.6.0', 'resnet50', pretrained=True
).cuda().half().eval()

# Errors out
model_opt = torch.compile(model, backend="torch_tensorrt", dynamic=False, options={"debug": True, "min_block_size": 1, "enabled_precisions": {torch.half}})

# Check correctness
torch.testing.assert_close(actual=model_opt(x), expected=model(x), rtol=1e-2, atol=1e-2)

# Benchmark
from hidet.utils import benchmark_func
print('eager: {:2f}'.format(benchmark_func(lambda: model(x))))
print('tensort-rt jit: {:2f}'.format(benchmark_func(lambda: model_opt(x))))