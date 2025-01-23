import hidet
import torch

# Load a pre-trained ResNet50 model
x = torch.randn(1, 3, 224, 224, device='cuda').half()
model = torch.hub.load(
    'pytorch/vision:v0.6.0', 'resnet50', pretrained=True
).cuda().half().eval()

# Configure hidet to use tensor core and enable tuning
hidet.torch.dynamo_config.use_tensor_core(True)
hidet.torch.dynamo_config.search_space(2) 

# Compile the model using Hidet
model_opt = torch.compile(model, backend='hidet')

# Check correctness
torch.testing.assert_close(actual=model_opt(x), expected=model(x), rtol=1e-2, atol=1e-2)

# Benchmark
from hidet.utils import benchmark_func
print('eager: {:2f}'.format(benchmark_func(lambda: model(x))))
print('hidet: {:2f}'.format(benchmark_func(lambda: model_opt(x))))


###
# eager: 10.526896
# hidet: 0.914621
###