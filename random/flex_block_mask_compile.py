import torch
from triton.testing import do_bench
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, noop_mask

torch.manual_seed(0)

import torch
torch.set_default_device('cuda')

def sliding_window(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() < 2048

S = 16384
create_block_mask_compiled = torch.compile(create_block_mask)

print("compiled flag: ", do_bench(lambda: create_block_mask(sliding_window, B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True)))
print("compiled: ", do_bench(lambda: create_block_mask_compiled(sliding_window, B=None, H=None, Q_LEN=S, KV_LEN=S)))
print("eager: ", do_bench(lambda: create_block_mask(sliding_window, B=None, H=None, Q_LEN=S, KV_LEN=S)))

# compiled flag:  1.4104219675064087
# compiled:  0.1276228427886963
# eager:  5.710970878601074