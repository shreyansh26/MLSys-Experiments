# WIP - Finding CUDA Aware mpi4py installation to be an issue
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from distributed_sampler import DistributedSampler

from mpi4py import MPI

def dist_setup():
    # works with torchrun
    local_rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    print(local_rank, world_size)

    return local_rank, world_size

def sync_initial_weights(model, local_rank, world_size):
    for param in model.parameters():
        if local_rank == 0:
            # Rank 0 is sending it's own weight
            # to all it's siblings (1 to world_size)
            for sibling in range(1, world_size):
                MPI.Comm.send(param.data, dst=sibling)
        else:
            # Siblings must recieve the parameters
            MPI.Comm.recv(param.data, source=0)

def sync_initial_weights_broadcast(model, local_rank, world_size):
    for param in model.parameters():
        param.data = MPI.Comm.bcast(param.data, root=0)

def reduce_gradients(model, local_rank, world_size):
    for param in model.parameters():
        MPI.Comm.Allreduce(sendbuf=param.grad, op=MPI.SUM)
        MPI.Comm.Barrier()
        param.grad /= world_size

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, local_rank, world_size, train_loader, optimizer, epoch):
    total_loss = torch.tensor([0], dtype=torch.float32).to(local_rank)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(local_rank), target.to(local_rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        reduce_gradients(model, local_rank, world_size)

        optimizer.step()

        total_loss += loss

        if batch_idx % args.log_interval == 0:
            loss = MPI.Comm.allreduce(loss, op=MPI.SUM)
            MPI.Comm.Barrier()
            logging_step_loss = loss / world_size
            if local_rank == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data) * world_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), logging_step_loss.item()))
                if args.dry_run:
                    break

    total_loss = MPI.Comm.allreduce(total_loss, op=MPI.SUM)
    MPI.Comm.Barrier()
    train_epoch_loss = total_loss / len(train_loader)
    train_epoch_loss = train_epoch_loss / world_size
    
    if local_rank == 0:
        print(f"Loss after epoch {epoch}: {train_epoch_loss.item()}")

def test(model, local_rank, world_size, test_loader):
    test_loss = torch.tensor([0], dtype=torch.float32).to(local_rank)
    correct = torch.tensor([0]).long().to(local_rank)

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(local_rank), target.to(local_rank)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='mean')  # average of batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss = MPI.Comm.allreduce(test_loss, op=MPI.SUM)
    MPI.Comm.Barrier()
    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_loss = test_epoch_loss / world_size

    correct = MPI.Comm.allreduce(correct, op=MPI.SUM)
    MPI.Comm.Barrier()

    if local_rank == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_epoch_loss.item(), correct.item(), len(test_loader.dataset),
            100. * correct.item() / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='For Saving the final Model from all ranks after training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    local_rank, world_size = dist_setup()

    if world_size > 0:
        if args.batch_size % world_size == 0 and args.test_batch_size % world_size == 0:
            train_batch_size = args.batch_size // world_size
            test_batch_size = args.test_batch_size // world_size
        else:
            print("Error in batch size and world rank compatibility")
            sys.exit(1)
    else:
        train_batch_size = args.batch_size
        test_batch_size = args.test_batch_size

    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if local_rank == 0:
        print("Training batch size:", train_kwargs['batch_size'])

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset_train = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset_test = datasets.MNIST('./data', train=False,
                       transform=transform)

    train_distributed_sampler = DistributedSampler(dataset_train, local_rank, world_size, shuffle=False) # To check for correctness with single-node train_mnist
    test_distributed_sampler = DistributedSampler(dataset_test, local_rank, world_size)

    train_kwargs.update({'sampler': train_distributed_sampler})
    test_kwargs.update({'sampler': test_distributed_sampler})

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = Net().to(local_rank)
    sync_initial_weights_broadcast(model, local_rank, world_size)
    MPI.Comm.Barrier()

    if args.debug:
        torch.save(model.state_dict(), f"models/initial_mnist_cnn_rank_{local_rank}.pt")

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, local_rank, world_size, train_loader, optimizer, epoch)
        MPI.Comm.Barrier()
        test(model, local_rank, world_size, test_loader)
        scheduler.step()

    if args.debug:
        torch.save(model.state_dict(), f"models/final_mnist_cnn_rank_{local_rank}.pt")

    if local_rank == 0 and args.save_model:
        torch.save(model.state_dict(), "models/mnist_cnn.pt")


if __name__ == '__main__':
    main()