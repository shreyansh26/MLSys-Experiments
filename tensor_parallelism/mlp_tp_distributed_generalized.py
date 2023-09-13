import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from mlp_tp_verify import MLP

def run(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, rank: int, size: int, ref: torch.Tensor=None):
    device = torch.device("cuda:{}".format(rank))

    A_split = torch.zeros(A.shape[0], A.shape[1] // size).to(device) # d * (4d/size)
    B_split = torch.zeros(B.shape[0] // size, B.shape[1]).to(device) # (4d/size) * d
    out = torch.zeros_like(x).to(device) # n x d

    req = None

    if rank == 0:
        device = torch.device("cuda:{}".format(rank))
        x = x.to(device)
        A = A.to(device)
        B = B.to(device)

        A_chunks = torch.chunk(A, size, dim=-1)
        # Send the splits to different devices
        for dst in range(size):
            if dst == 0:
                continue
            req = dist.isend(tensor=A_chunks[dst].contiguous(), dst=dst)
            req.wait()
            print(f'Rank 0 started sending to rank {dst}')

        A_split = A_chunks[0]
        XA = torch.matmul(x, A_split) 
        Y = nn.GELU()(XA)

        B_chunks = torch.chunk(B, size, dim=0)
        # Send the splits to different devices
        for dst in range(size):
            if dst == 0:
                continue
            req = dist.isend(tensor=B_chunks[dst].contiguous(), dst=dst)
            req.wait()
            print(f'Rank 0 started sending to rank {dst}')
        
        B_split = B_chunks[0]
        out = torch.matmul(Y, B_split)
        
    else:
        device = torch.device("cuda:{}".format(rank))
        x = x.to(device)

        # Receive tensor from process 0
        req = dist.irecv(tensor=A_split, src=0)
        print('Rank 1 started receiving A_split')
        req.wait()

        # Now we have A_split
        XA = torch.matmul(x, A_split) 
        Y = nn.GELU()(XA)

        # Receive tensor from process 0
        req = dist.irecv(tensor=B_split, src=0)
        print('Rank 1 started receiving B_split')
        req.wait()

        out = torch.matmul(Y, B_split)
        
    req.wait()
    print("Completed", req.is_completed())
    group = dist.new_group(list(range(size)))
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
    d = 51200
    mlp = MLP(d)
    x = torch.rand(n, d)
    A = mlp.A
    B = mlp.B

    out1 = None
    # out1 = mlp.direct_forward(x)
    # out2 = mlp.split_forward(x)
    # print(torch.allclose(out1, out2))
    

    size = 4
    processes = []
    mp.set_start_method("spawn")

    for rank in range(size):
        p = mp.Process(target=init_process, args=(x, A, B, rank, size, run, out1))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()