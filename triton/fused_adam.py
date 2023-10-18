import copy
import torch
import torch.nn as nn

import triton
import triton.language as tl

@triton.jit
def fused_adam_kernel(
    params_ptr,
    grads_ptr,
    n_ele,
    m_ptr,
    v_ptr,
    lr, 
    beta1, beta2,
    beta1_pow_step, beta2_pow_step,
    eps,
    wd,
    step_count,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    grads += wd * params

    m_new = beta1 * m + (1 - beta1) * grads
    v_new = beta2 * v + (1 - beta2) * (grads * grads)

    m_new_corrected = m_new / (1 - beta1_pow_step)
    v_new_corrected = v_new / (1 - beta2_pow_step)

    params_new = params - (lr * m_new_corrected / (tl.sqrt(v_new_corrected) + eps))

    tl.store(params_ptr + offsets, params_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

class AdamFused:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        self.parameters = list(parameters)
        self.n_ele = sum(param.numel() for param in self.parameters)
        self.params = None
        self.grads = None
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.step_count = 0
        
        self.init_params_and_grads()
        self.init_moments()

    def init_params_and_grads(self):
        self.params = torch.zeros(self.n_ele, dtype=self.parameters[0].dtype, device=self.parameters[0].device)
        self.grads = torch.zeros(self.n_ele, dtype=self.parameters[0].dtype, device=self.parameters[0].device)

        i = 0
        for param in self.parameters:
            num_ele =  param.numel()
            # Populate self.params list
            self.params[i : i+num_ele] = param.view(-1)
            # Ensure that original model will be updated 
            # on updating self.params
            param.data = self.params[i : i+num_ele].view(param.data.shape)
            param.grad = self.grads[i : i+num_ele].view(param.data.shape)

            i += num_ele

        self.params.grad = self.grads

    def init_moments(self):
        self.m = torch.zeros_like(self.params)
        self.v = torch.zeros_like(self.params)

    def zero_grad(self, set_to_none=False):
        if self.params.grad is not None:
            if set_to_none:
                self.params.grad = None
            else:
                if self.params.grad.grad_fn is not None:
                    self.params.grad.detach_()
                else:
                    self.params.grad.requires_grad_(False)
                self.params.grad.zero_()

    def step(self):
        self.step_count += 1

        with torch.no_grad():
            grid = lambda meta: (triton.cdiv(self.n_ele, meta['BLOCK_SIZE']), )

            fused_adam_kernel[grid](self.params, self.grads, self.n_ele, self.m, self.v, self.lr, self.beta1, self.beta2, self.beta1 ** self.step_count, self.beta2 ** self.step_count, self.eps, self.wd, self.step_count, BLOCK_SIZE=1024)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device == torch.device('cuda')

def check_allclose(tensor1, tensor2):
    if torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-5): # Not an EXACT match so have to lower constraints
        print("FusedAdam oputput same as Adam!")
    else:
        print("Outputs differ...")

def verify_same_model(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1.data, p2.data, rtol=1e-3, atol=1e-5):
            return False
    return True

def train_dummy(model, data, opt):
    for x in data:
        opt.zero_grad()
        loss = model(x).sum()
        loss.backward()
        opt.step()

def test_adam():
    test_cases = [
        dict(lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0),
        dict(lr=0.1, betas=(0.8, 0.9), eps=1e-8, weight_decay=0.05),
        dict(lr=0.2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01),
    ]
    for config in test_cases:
        data = torch.randn(10, 10, 10).to(device)
    
        torch.manual_seed(1023)
        model1 = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 10)).to(device)
        model2 = copy.deepcopy(model1)

        opt_adam = torch.optim.Adam(model1.parameters(), **config)
        train_dummy(model1, data, opt_adam)
    
        opt_fused_adam = AdamFused(model2.parameters(), **config)
        train_dummy(model2, data, opt_fused_adam)
    
        print("Config: ", config)
        if verify_same_model(model1, model2):
            print("✅ AdamFused oputput same as Adam!")
        else:
            print("❌ Outputs differ...")
        print("***"*20)
    
test_adam()

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # Argument names to use as an x-axis for the plot
        xlabel="Number of data points",
        x_vals=[
            2**i for i in range(13)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['adam-torch', 'fused-adam-triton'],
        # Label name for the lines
        line_names=["Adam (PyTorch)", "Adam-Fused (Triton)"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="Time to Train (ms)",  # Label name for the y-axis
        plot_name="fused-adam",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(N, provider):
    config = dict(lr=0.2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 10)).to(device)
    data = torch.randn(N, 10, 10).to(device)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'adam-torch':
        opt = torch.optim.Adam(model.parameters(), **config)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: train_dummy(model, data, opt), percentiles=quantiles)
    if provider == 'fused-adam-triton':
        opt = AdamFused(model.parameters(), **config)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: train_dummy(model, data, opt), percentiles=quantiles)
    return ms, max_ms, min_ms


benchmark.run(print_data=True, save_path='plots/fused-adam')