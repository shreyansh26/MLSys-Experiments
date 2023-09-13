import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from mlp_tp_verify import MLP

def run(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, rank: int, size: int, ref: torch.Tensor=None):
    device = torch.device("cuda:{}".format(rank))

    A2 = torch.zeros(x.shape[1], x.shape[1] * 2).to(device) # d * 2d
    B2 = torch.zeros(x.shape[1] * 2, x.shape[1]).to(device) # 2d * d
    out = torch.zeros_like(x).to(device) # n x d

    req = None

    if rank == 0:
        device = torch.device("cuda:{}".format(rank))
        x = x.to(device)
        A = A.to(device)
        B = B.to(device)
        A1, A2 = torch.chunk(A, 2, dim=-1)
        # Send the tensor to process 1
        req = dist.isend(tensor=A2.contiguous(), dst=1)
        req.wait()
        print('Rank 0 started sending A2')

        XA1 = torch.matmul(x, A1) 
        Y1 = nn.GELU()(XA1)

        B1, B2 = torch.chunk(B, 2, dim=0)
        # Send the tensor to process 1 
        req = dist.isend(tensor=B2.contiguous(), dst=1)
        req.wait()
        print('Rank 0 started sending B2')
        
        out = torch.matmul(Y1, B1)
        
    else:
        device = torch.device("cuda:{}".format(rank))
        x = x.to(device)

        # Receive tensor from process 0
        req = dist.irecv(tensor=A2, src=0)
        print('Rank 1 started receiving A2')
        req.wait()

        # Now we have A2
        XA2 = torch.matmul(x, A2) 
        Y2 = nn.GELU()(XA2)

        # Receive tensor from process 0
        req = dist.irecv(tensor=B2, src=0)
        print('Rank 1 started receiving B2')
        req.wait()

        out = torch.matmul(Y2, B2)
        
    req.wait()
    print("Completed", req.is_completed())
    group = dist.new_group([0, 1])
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    
    if ref is not None and rank == 0:
        device = torch.device("cuda:{}".format(rank))
        print(torch.allclose(out, ref.to(device)))

def init_process(x, A, B, rank, size, fn, ref=None, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(x, A, B, rank, size, ref)


if __name__ == "__main__":
    n = 1000
    d = 16384
    mlp = MLP(d)
    x = torch.rand(n, d)
    A = mlp.A
    B = mlp.B

    out1 = None
    # out1 = mlp.direct_forward(x)
    # out2 = mlp.split_forward(x)
    # print(torch.allclose(out1, out2))
    

    size = 2
    processes = []
    mp.set_start_method("spawn")

    for rank in range(size):
        p = mp.Process(target=init_process, args=(x, A, B, rank, size, run, out1))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()