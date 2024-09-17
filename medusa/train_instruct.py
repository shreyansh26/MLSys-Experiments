import random
from core.supervised_dataset import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    SupervisedDataset,
    DataCollatorForSupervisedDataset,
)
from medusa_util import add_medusa_heads
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datetime import datetime
from safetensors.torch import save_file
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    _or_policy, lambda_auto_wrap_policy
)
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from liger_kernel.transformers import apply_liger_kernel_to_mistral
from core.multipack_sampler import MultipackDistributedBatchSampler
from dotenv import load_dotenv
import functools
import torch.distributed as dist
import wandb
import uuid
import torch
import transformers
import os
import math
import numpy as np
from medusa_util import ResBlock
from torch.nn import Linear, Sequential

load_dotenv()


def disable_model_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def setup_model(model_name, max_length):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        use_auth_token=os.environ["HF_TOKEN"],
    )
    config.sliding_window = 4096
    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=os.environ["HF_TOKEN"],
        config=config,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        use_auth_token=os.environ["HF_TOKEN"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Vocab size:", tokenizer.vocab_size)

    # special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def evaluation(
    model,
    eval_dataloader,
    wandb,
    local_rank,
):
    if local_rank == 0:
        print("RUNNING EVAL")

    model.eval()
    losses = 0
    for step, batch in enumerate(eval_dataloader):
        inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "labels": batch["labels"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss
        losses += loss.float()

    losses = losses / (step + 1)
    val_loss = get_all_reduce_mean(losses.clone()).item()

    if local_rank == 0:
        print(f"Validation Loss {val_loss:.4f}")

    if local_rank == 0:
        wandb.log(
            {
                "val_loss": val_loss,
            }
        )

    return val_loss


def get_dataloader(
    use_multipack_sampler,
    max_length,
    dataset,
    world_size,
    local_rank,
    shuffle,
    seed,
    collator,
    batch_size,
):
    if use_multipack_sampler:
        lengths = np.array([len(tokens["input_ids"]) for tokens in dataset])
        sampler = MultipackDistributedBatchSampler(
            batch_max_length=batch_size * max_length,
            lengths=lengths,
            num_replicas=world_size,
            rank=local_rank,
            seed=seed,
        )

        loader = DataLoader(
            dataset,
            pin_memory=True,
            collate_fn=collator,
            batch_sampler=sampler,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=shuffle,
            seed=seed,
        )

        loader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator,
            sampler=sampler,
        )

    return sampler, loader


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, lr, weight_decay):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def should_run_eval(total_steps, times_to_run, current_step):
    return current_step % (total_steps // times_to_run) == 0


def log_stats(pbar, wandb, epoch, loss_tensor, grad_norm, scheduler, step_size):
    last_lr = scheduler.get_last_lr()[0]

    wandb.log(
        {
            "current_loss": loss_tensor,
            "current_epoch": epoch,
            "learning_rate": last_lr,
            "grad_norm": grad_norm,
        },
    )

    current_loss = f"{loss_tensor:.4f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")
    pbar.update(step_size)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_warmup_steps(num_training_steps, warmup_ratio=0.05):
    return math.ceil(num_training_steps * warmup_ratio)


def clip_model_gradients(model, max_grad_norm):
    # return model.clip_grad_norm_(max_grad_norm).item()
    return torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_grad_norm).item()


def get_scheduler(local_rank, scheduler_type, optimizer, max_steps):
    warmup_steps = get_warmup_steps(max_steps)

    if local_rank == 0:
        print(f"[WARMUP STEPS]: {warmup_steps}")
        print(f"[MAX STEPS]: {max_steps}")
        print(f"[SCHEDULER]: {scheduler_type}")

    return transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )


def save_model(local_rank, model, tokenizer, medusa_return, medusa_only_heads, outpath, current_epoch, current_step):
    if medusa_return and medusa_only_heads:
        if hasattr(model, "module"):
            lm_head = model.module.medusa_head
        else:
            lm_head = model.medusa_head

        # save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        #     cpu_state = lm_head.state_dict()
        cpu_state = lm_head.state_dict()

        if local_rank == 0:
            print(cpu_state)
            print(f"SAVING LM Heads")
            outpath += f"/epoch_{current_epoch}/step_{current_step}"
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            save_file(
                    cpu_state,
                    os.path.join(outpath, "model.safetensors"),
                )
    else:
        if hasattr(model, "module"):
            model = model.module
        
        cpu_state = model.state_dict()

        if local_rank == 0:
            print(cpu_state)
            print(f"SAVING LM Heads")
            outpath += f"/epoch_{current_epoch}/step_{current_step}"
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            save_file(
                    cpu_state,
                    os.path.join(outpath, "model.safetensors"),
                )

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    print("Local rank", local_rank)
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    scheduler_type = "cosine"
    seed = 877645  # set your seed
    transformers.set_seed(seed)

    run_id = str(uuid.uuid4())
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")
    max_length = 4096  # adjust as needed
    disable_dropout = False
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True  # multipack sampler already does random sampling
    train_batch_size = 4  # adjust as needed
    validation_batch_size = 4  # adjust as needed
    epochs = 10  # adjust as needed
    gradient_accumulation_steps = 4
    acc_steps = 0  # TODO: not implemented here yet
    lr = 3e-4  # adjust as needed
    weight_decay = 0.01  # adjust as needed
    gradient_clipping = 1.0  # adjust as needed
    train_on_inputs = False  # whether to train on instruction tokens
    use_multipack_sampler = (
        False  # whether to use the multipack sampler or torch sampler
    )
    medusa_num_layers = 1
    medusa_num_heads = 3
    medusa_return = True
    medusa_only_heads = True
    output_dir = f"medusa_mistral/{lr}_{medusa_num_layers=}_{medusa_only_heads=}_{medusa_num_heads=}_{train_batch_size}_{date_of_run}"

    model, tokenizer = setup_model(model_name, max_length)
    model.config.pretraining_tp = 1
    apply_liger_kernel_to_mistral()

    # Freeze the base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    add_medusa_heads(
        model,
        medusa_num_heads,
        medusa_num_layers,
        medusa_return,
        medusa_only_heads,
    )

    model = model.to(torch.cuda.current_device())
    print(model.forward)

    print(model.medusa_head)
    print(model.medusa_head.state_dict())

    for k, v in model.medusa_head.state_dict().items():
        print(k, v, v.shape)
        print("**")

    num_params = sum([p.numel() for p in model.parameters()])

    # def custom_auto_wrap_policy(
    #     module: torch.nn.Module,
    #     recurse: bool,
    #     nonwrapped_numel: int,
    #     *,  # Force keyword arguments
    #     min_num_params: int = 1e8,
    #     medusa_head_module: torch.nn.Module = None,
    #     fsdp_root_module: torch.nn.Module = None
    # ):
    #     # Do not wrap if recursing or if the module is the root module
    #     if recurse or module is fsdp_root_module:
    #         return False
    #     if module is medusa_head_module:
    #         return True
    #     if isinstance(module, MistralDecoderLayer):
    #         return True
    #     return False

    # # After defining your model and adding medusa heads
    # medusa_head_module = model.medusa_head  # Reference to the medusa_head module
    # fsdp_root_module = model

    # auto_wrap_policy = functools.partial(
    #     custom_auto_wrap_policy,
    #     medusa_head_module=medusa_head_module,
    #     fsdp_root_module=fsdp_root_module
    # )

    # # auto_wrap_policy = functools.partial(
    # #     transformer_auto_wrap_policy,
    # #     transformer_layer_cls={
    # #         MistralDecoderLayer,
    # #         Sequential
    # #     },
    # # )

    # fsdp_config = dict(
    #     # auto_wrap_policy=auto_wrap_policy,
    #     sharding_strategy=ShardingStrategy.FULL_SHARD,
    #     device_id=torch.cuda.current_device(),
    #     mixed_precision=MixedPrecision(
    #         param_dtype=torch.bfloat16,
    #         reduce_dtype=torch.bfloat16,
    #         buffer_dtype=torch.bfloat16,
    #     ),
    #     sync_module_states=True,
    #     backward_prefetch=None,
    #     param_init_fn=None,
    #     cpu_offload=None,
    #     use_orig_params=True
    # )

    # # model = FSDP(model, **fsdp_config)
    # # model.medusa_head = FSDP(model.medusa_head, **fsdp_config)

    # Figure out how to make FSDP training work
    model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    # with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    medusa_state_dict = model.module.medusa_head.state_dict()
    print("**"*100)
    print(medusa_state_dict)

    for k, v in medusa_state_dict.items():
        print(k, v, v.shape)
        print("**")

    optimizer = get_optimizer(model, lr, weight_decay)
 
    train_ds = ["train_json.jsonl"]
    val_ds = ["val_json.jsonl"]

    train_dataset = SupervisedDataset(train_on_inputs, tokenizer, train_ds,
                                use_instruct_format=True)
    val_dataset = SupervisedDataset(train_on_inputs, tokenizer, val_ds,
                                use_instruct_format=True)
    collator = DataCollatorForSupervisedDataset(tokenizer)

    train_sampler, train_loader = get_dataloader(
        use_multipack_sampler,
        max_length,
        train_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        train_batch_size,
    )
    val_sampler, val_loader = get_dataloader(
        use_multipack_sampler,
        max_length,
        val_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        validation_batch_size,
    )

    if use_multipack_sampler:
        total_steps_per_epoch = train_sampler.num_batches()
    else:
        total_steps_per_epoch = len(train_loader)

    max_steps = total_steps_per_epoch * epochs
    scheduler = get_scheduler(local_rank, scheduler_type, optimizer, max_steps)

    if local_rank == 0:
        run = wandb.init(
            project="mistral-7b-v2",
            name=run_id,
            config={
                "model_name": model_name,
                "run_id": run_id,
                "date": date_of_run,
                "dataset_size": len(train_dataset),
                "dataset": ",".join(train_ds),
                "validation": ",".join(val_ds),
                "weight_decay": weight_decay,
                "clip_gradients": clip_gradients,
                "learning_rate": lr,
                "shuffle": shuffle,
                "seed": seed,
                "disable_dropout": disable_dropout,
                "use_multipack_sampler": use_multipack_sampler,
                "train_on_inputs": train_on_inputs,
                "epochs": epochs,
                "acc_steps": acc_steps,
                "batch_size": train_batch_size,
                "total_batch_size": train_batch_size * world_size,
                "scheduler_type": scheduler_type,
            },
        )

    # if gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    # if disable_dropout:
    #     disable_model_dropout(model)

    model.train()
    dist.barrier()

    train_iterator = iter(train_loader)
    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)
        current_epoch = epoch + 1

        pbar = tqdm(
            total=total_steps_per_epoch,
            colour="blue",
            desc=f"Epoch {epoch}.00",
            disable=(local_rank != 0),
        )
        current_step = 0
        while True:
            acc_loss = torch.tensor(0.).to(model.device)
            actual_accumulation_steps = 0
            for acc_step in range(gradient_accumulation_steps):
                model.require_backward_grad_sync = (acc_step == gradient_accumulation_steps - 1) or \
                                                    (current_step == total_steps_per_epoch - 1)
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                inputs = {
                        "input_ids": batch["input_ids"].to(model.device),
                        "labels": batch["labels"].to(model.device),
                        "attention_mask": batch["attention_mask"].to(model.device),
                    }
                # forward
                outputs = model(**inputs)
                loss = outputs.loss
                acc_loss += loss.item()
                loss /= gradient_accumulation_steps

                # backward
                loss.backward()
                
                current_step += 1
                actual_accumulation_steps += 1

                if current_step == total_steps_per_epoch:
                    break

            acc_loss /= actual_accumulation_steps
            
            # clipping
            if clip_gradients:
                grad_norm = clip_model_gradients(model, gradient_clipping)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # avg loss over all processes
            acc_loss = get_all_reduce_mean(acc_loss).item()

            if local_rank == 0:
                log_stats(
                    pbar,
                    wandb,
                    round((current_step / total_steps_per_epoch), 2) + epoch,
                    acc_loss,
                    grad_norm,
                    scheduler,
                    actual_accumulation_steps
                )

            if current_step == total_steps_per_epoch:
                validation_loss = evaluation(
                    model,
                    val_loader,
                    wandb,
                    local_rank,
                )

                save_model(
                    local_rank,
                    model,
                    tokenizer,
                    medusa_return, 
                    medusa_only_heads,
                    output_dir,
                    current_epoch,
                    current_step,
                )

                model.train()
                break

    # save final model
    save_model(local_rank, model, tokenizer, medusa_return, medusa_only_heads, output_dir, epochs, "final")