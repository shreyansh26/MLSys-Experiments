import torch_tensorrt
import torch

# Load a pre-trained ResNet50 model
x = torch.randn(1, 3, 224, 224, device='cuda').half()
model = torch.hub.load(
    'pytorch/vision:v0.6.0', 'resnet50', pretrained=True
).cuda().half().eval()

inputs = [torch_tensorrt.Input(
            min_shape=[1, 3, 224, 224],
            opt_shape=[1, 3, 512, 512],
            max_shape=[1, 3, 1024, 1024],
            dtype=torch.half)]
enabled_precisions = {torch.float16}

model_opt = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions)

# Check correctness
torch.testing.assert_close(actual=model_opt(x).half(), expected=model(x), rtol=1e-2, atol=1e-2)

# Benchmark
from hidet.utils import benchmark_func
print('eager: {:2f}'.format(benchmark_func(lambda: model(x))))
print('tensort-rt aot: {:2f}'.format(benchmark_func(lambda: model_opt(x))))

###
# eager: 6.415510
# tensort-rt aot: 0.828218
###