# Works after this (for now) - `pip install --pre torch torch-tensorrt --extra-index-url https://download.pytorch.org/whl/nightly/cu121`
import torch_tensorrt
import torch

# Load a pre-trained ResNet50 model
x = torch.randn(1, 3, 224, 224, device='cuda').half()
model = torch.hub.load(
    'pytorch/vision:v0.6.0', 'resnet50', pretrained=True
).cuda().half().eval()

# inputs = [torch_tensorrt.Input(
#             min_shape=[1, 3, 224, 224],
#             opt_shape=[1, 3, 512, 512],
#             max_shape=[1, 3, 1024, 1024],
#             dtype=torch.half)]
inputs = [torch_tensorrt.Input(
            shape=[1, 3, 224, 224],
            dtype=torch.half)]
enabled_precisions = {torch.half}

model_opt = torch.jit.script(model)
model_opt = torch_tensorrt.compile(model_opt, inputs=inputs, enabled_precisions=enabled_precisions, debug=True)

TOT = 100
cnt_fail = 0
for _ in range(100):
    x = torch.randn(1, 3, 224, 224, device='cuda').half()
    try:
        # Check correctness
        torch.testing.assert_close(actual=model_opt(x).half(), expected=model(x), rtol=1e-2, atol=1e-2)
    except AssertionError as e:
        print(e)
        cnt_fail += 1

# torch.half may have errors but torch.float32 works fine
print(f"Success Rate = {(TOT - cnt_fail)/TOT * 100}")


# Benchmark
from hidet.utils import benchmark_func
print('eager: {:2f}'.format(benchmark_func(lambda: model(x))))
print('tensort-rt aot: {:2f}'.format(benchmark_func(lambda: model_opt(x))))

###
# eager: 6.763744
# tensort-rt aot: 0.630522
###