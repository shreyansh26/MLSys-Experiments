# TORCH_LOGS="recompiles" python recompile_int.py
import torch

@torch.compile
def fn(x, c):
    return x + c

for i in range(1, 10):
    fn(torch.ones(i), 0.5 + i)

'''
$ TORCH_LOGS="recompiles" python playground.py
Recompiling function fn in /data/users/williamwen/pytorch/playground.py:3
    triggered by the following guard failure(s):
    - 0/7: L['c'] == 8.5
    - 0/6: L['c'] == 7.5
    - 0/5: L['c'] == 6.5
    - 0/4: L['c'] == 5.5
    - 0/3: L['c'] == 4.5
    - 0/2: L['c'] == 3.5
    - 0/1: L['c'] == 2.5
    - 0/0: L['c'] == 1.5
torch._dynamo hit config.cache_size_limit (8)
    function: 'fn' (/data/users/williamwen/pytorch/playground.py:3)
    last reason: 0/0: L['c'] == 1.5
----

import torch

mod = torch.nn.Linear(3, 3)
opt = torch.optim.Adam(mod.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)

@torch.compile
def fn(inp):
    opt.zero_grad(True)
    out = mod(inp).sum()
    out.backward()
    opt.step()
    sched.step()

for i in range(1, 10):
    fn(torch.ones(3, 3))

$ TORCH_LOGS="recompiles" python playground.py
Recompiling function step in /data/users/williamwen/pytorch/torch/optim/adam.py:189
    triggered by the following guard failure(s):
    - 3/7: L['self'].param_groups[0]['lr'] == 0.004782969000000002
    - 3/6: L['self'].param_groups[0]['lr'] == 0.005314410000000002
    - 3/5: L['self'].param_groups[0]['lr'] == 0.005904900000000002
    - 3/4: L['self'].param_groups[0]['lr'] == 0.006561000000000002
    - 3/3: L['self'].param_groups[0]['lr'] == 0.007290000000000001
    - 3/2: L['self'].param_groups[0]['lr'] == 0.008100000000000001
    - 3/1: L['self'].param_groups[0]['lr'] == 0.009000000000000001
    - 3/0: L['self'].param_groups[0]['lr'] == 0.01
torch._dynamo hit config.cache_size_limit (8)
    function: 'step' (/data/users/williamwen/pytorch/torch/optim/adam.py:189)
    last reason: 3/0: L['self'].param_groups[0]['lr'] == 0.01
---

In both examples, we can wrap float variables in tensors in order to prevent recompilations.

# I don't find the above outputs when running the code - But still a good point to wrap int and float in tensors.

# first example
for i in range(1, 10):
    fn(torch.ones(i), torch.tensor(0.5 + i))

# second example
opt = torch.optim.Adam(mod.parameters(), lr=torch.tensor(0.01))
sched = torch.optim.lr_scheduler.ExponentialLR(opt, torch.tensor(0.9))
'''