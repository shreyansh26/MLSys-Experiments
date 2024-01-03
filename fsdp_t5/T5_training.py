# CUDA_VISIBLE_DEVICES=0,3 torchrun --nnodes 1 --nproc_per_node 2 T5_training.py
# CUDA_VISIBLE_DEVICES=5,7 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=105 --rdzv_endpoint="0.0.0.0:29505" T5_training.py
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast, DataCollatorForSeq2Seq
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from tokenizers import AddedToken
import functools
from torch.optim.lr_scheduler import StepLR, LambdaLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.t5.modeling_t5 import T5Block

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
 checkpoint_wrapper,
 CheckpointImpl)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from summarization_dataset import *
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
import tqdm
from datetime import datetime
from data_utils import get_datasets

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

g_gigabyte = 1024**3

def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

bfSixteen_mixed = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, T5Block)

def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
  
    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        # print(batch)
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy


def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def setup_model(model_name, max_token_count):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_cache=False)
    # from optimum.bettertransformer import BetterTransformer
    # model = BetterTransformer.transform(model) 
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=max_token_count)
    # Optional - required for JSON outputs
    tokenizer.add_tokens([AddedToken("\n", normalized=False)])
    tokenizer.add_tokens([AddedToken("{", normalized=False)])
    tokenizer.add_tokens([AddedToken("}", normalized=False)])
    tokenizer.add_tokens([AddedToken("\\", normalized=False)])

    # model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def fsdp_main(args):

    model, tokenizer = setup_model("google/flan-t5-xl", 2048)

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    tokenized_dataset = get_datasets("data/train_sample.csv", "data/val_sample.csv", tokenizer, "google/flan-t5-xl", 2048)
    train_dataset, val_dataset = tokenized_dataset["train"], tokenized_dataset["valid"]

    if rank == 0:
        print(train_dataset['labels'][0])

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    setup()

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1, 'collate_fn': data_collator}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2, 'collate_fn': data_collator}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    
    for x in train_loader:
        if rank == 0:
            print("Train Loader batch length:", x["input_ids"].shape)
        break

    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD #SHARD_GRAD_OP for Zero2 and FULL_SHARD for Zero3
    cpu_offload: CPUOffload = CPUOffload(offload_params=True)
    torch.cuda.set_device(local_rank)
   
   
    #init_start_event = torch.cuda.Event(enable_timing=True)
    #init_end_event = torch.cuda.Event(enable_timing=True)

    #init_start_event.record()
    
    bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and dist.is_nccl_available()
    )
    
    if bf16_ready:
        mp_policy = bfSixteen
    else:
        mp_policy = None # defaults to fp32
        
    if rank == 0:
        print("MP Policy", mp_policy)
        # print("Model N-positions:", model.config.n_positions)
        print("Tokenizer max-length:", tokenizer.model_max_length)
        print("Tokenizer vocab length:", len(tokenizer))
        print("Model encoder embed shape:", model.encoder.embed_tokens.weight.shape)
        print("Model decoder embed shape:", model.decoder.embed_tokens.weight.shape)
        print("Model lm_head shape:", model.lm_head.weight.shape)

    model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        # cpu_offload=cpu_offload,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=False)

    apply_fsdp_checkpointing(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.epochs)

    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "models/T5-model-"

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        if args.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()
        
        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print(f"completed save and stats zone...")
        

    #init_end_event.record()
        
    #if rank == 0:
        #print(f"Cuda event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        # print(f"{model}")

        # if args.save_model and curr_val_loss < best_val_loss:
        if args.save_model:
            # save
            if rank == 0:
                print(f"--> entering save model state")
            
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            #print(f"saving process: rank {rank}  done w state_dict")
           

            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                print(f"--> attempting to save model prefix {currEpoch}")
                save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                print(f"--> saving as model name {save_name}")

                torch.save(cpu_state, save_name)
            
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: .002)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    fsdp_main(args)
