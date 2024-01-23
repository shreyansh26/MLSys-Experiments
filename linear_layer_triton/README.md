# Linear Layer using Triton

**This is an experimental repository, things may break.**

The main aim of this repository is to understand the low-level improvements that tools like `torch.compile` provide and implement some of it myself.

This repository contains -
* A LinearLayer (nn.LinearLayer replacement) in Triton (both forward and backward pass) [src/](src/)
* Also includes a fusion step wherein an activation (optional) is within the Triton kernel [src/kernels](src/kernels)
* Automated patching of nn.LinearLayer layers to the new Triton layers [src/patch_linear_layer.py](src/patch_linear_layer.py)
* Benchmarking and testing [test/](test/)
* Examples of using the custom LinearLayer for training (custom DNN for MNIST and FlanT5-Base on Samsum) and inference (FLanT5-Base) [examples/](examples/)
* Optimized patching using CUDA Graphs (only for inference for now) [misc/patch_model.py](misc/patch_model.py)

On training of FlanT5-Base on Samsum dataset, using the Triton LinearLayer (with fusion) results in upto 1.6x speedup.
On inference of FlanT5-Base, using the Triton LinearLayer (with fusion) results in upto 3x speedups when used with CUDA Graphs.

## Credits
The [kernl](https://github.com/ELS-RD/kernl) project has been an inspiration for this deep-dive and I learnt quite a bit from their code as well.
