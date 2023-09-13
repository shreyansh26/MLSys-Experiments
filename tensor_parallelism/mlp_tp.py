# Forward pass only for now
# torchrun --nnodes=1 --nproc-per-node=4 mlp_tp.py -n 100 -d 4096
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import argparse

def dist_setup():
    # works with torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(local_rank, world_size)

    dist.init_process_group(
                backend="nccl",
                init_method='env://',
                world_size=world_size,
                rank=local_rank
            )
    return local_rank, world_size

class MLP(nn.Module):
    def __init__(self, d, world_size):
        super().__init__()
        self.output_dim = (4 * d) // world_size
        self.A = nn.Linear(d, self.output_dim, bias=False)
        self.B = nn.Linear(self.output_dim, d, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        torch.save(self.A.weight, f'artifacts/A_{local_rank}.pt')
        torch.save(self.B.weight, f'artifacts/B_{local_rank}.pt')
        x = self.A(x)
        x = self.gelu(x)
        x = self.B(x)
        return x

def run(X, local_rank, world_size):
    X = X.to(local_rank)
    mlp = MLP(d, world_size)
    mlp = mlp.to(local_rank)

    out = mlp(X)

    dist.all_reduce(out, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        print(out)
        torch.save(out, f'artifacts/out_{local_rank}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", required=True, type=int, help="N")
    parser.add_argument("-d", required=True, type=int, help="D")
    args = parser.parse_args()

    n = args.n
    d = args.d

    local_rank, world_size = dist_setup()

    # Setting seed ensures same input for all workers
    np.random.seed(0)
    X = torch.tensor(np.random.normal(0, 0.1, (n, d)), dtype=torch.float32)

    torch.save(X, f'artifacts/input_{local_rank}.pt')

    run(X, local_rank, world_size)