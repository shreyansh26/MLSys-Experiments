import math
import torch
from torch.utils.data import Sampler

class DistributedSampler(Sampler):
    def __init__(self, dataset, rank, world_size, shuffle=False, seed=0, drop_last=False):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.world_size != 0:
            self.num_samples_per_shard = math.ceil((len(self.dataset) - self.world_size) / self.world_size)
        else:
            self.num_samples_per_shard = math.ceil(len(self.dataset) / self.world_size)

        self.dataset_size = self.num_samples_per_shard * self.world_size
        
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.drop_last is False:
            # add extra examples to make it divisible
            padding_size = self.dataset_size - len(indices)

            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.dataset_size]

        assert len(indices) == self.dataset_size

        # Shard
        indices = indices[self.rank : self.dataset_size : self.world_size]

        assert len(indices) == self.num_samples_per_shard

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_shard

    def set_epoch(self, epoch):
        self.epoch = epoch