import argparse
import math
from pathlib import Path
from typing import Any
from datetime import datetime
import pytz

import torch

wandb_import_error: Exception | None = None
try:
    import wandb  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    wandb = None
    wandb_import_error = exc
from peft import LoraConfig, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import DataConfig, build_dataloaders
from utils import create_optimizer, create_scheduler, set_random_seed

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_OUTPUT_ROOT = "/mnt/ssd2/shreyansh/models/multilora"

def get_ist_time():
    # Return current time in IST
    return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d_%H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Llama 3B Instruct")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset identifier as defined in local_dataset.py")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Base model name or path")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Base directory for saving checkpoints")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma separated module names for LoRA injection",
    )
    parser.add_argument("--use-bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--eval-interval", type=int, default=0, help="Number of steps between eval during training (0 = eval each epoch)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile with dynamic shapes")
    return parser.parse_args()


def init_wandb(run_args: argparse.Namespace, metadata: dict[str, Any], lora_modules: list[str], current_time: str) -> None:
    if wandb is None:
        raise ImportError(
            "wandb is required but not installed. Please install it in the target environment"
        ) from wandb_import_error
    wandb.init(
        project="multilora",
        name=run_args.dataset_name + "_" + current_time,
        config={
            "dataset": run_args.dataset_name,
            "model": run_args.model_name,
            "batch_size": run_args.batch_size,
            "learning_rate": run_args.learning_rate,
            "weight_decay": run_args.weight_decay,
            "warmup_ratio": run_args.warmup_ratio,
            "num_epochs": run_args.num_epochs,
            "gradient_accumulation_steps": run_args.gradient_accumulation_steps,
            "max_samples": run_args.max_samples,
            "max_length": run_args.max_length,
            "train_size": metadata["train_size"],
            "eval_size": metadata["eval_size"],
            "lora_target_modules": lora_modules,
        },
    )


def prepare_model(args: argparse.Namespace) -> tuple[Any, Any]:
    torch_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[module.strip() for module in args.lora_target_modules.split(",")],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    if args.use_bf16:
        model.to(dtype=torch.bfloat16)
    if getattr(args, "compile", False):
        # dynamic shapes to avoid recompiles for varying sequence lengths
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)

    return tokenizer, model


def run_training(args: argparse.Namespace) -> None:
    current_time = get_ist_time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)
    if device.type == "cuda":
        # accelerate any residual fp32 GEMMs with TF32; bf16 path is unaffected
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer, model = prepare_model(args)
    model.print_trainable_parameters()

    data_config = DataConfig(
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    train_loader, eval_loader, metadata = build_dataloaders(tokenizer, data_config)

    lora_modules = [name for name in model.peft_config["default"].target_modules]
    print(f"LoRA modules: {lora_modules}")
    init_wandb(args, metadata, lora_modules, current_time)

    model = model.to(device)

    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_training_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_epochs
    num_warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = create_scheduler(optimizer, num_warmup_steps, total_training_steps)

    output_dir = Path(args.output_root) / f"{args.dataset_name}_{current_time}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch + 1}")
        latest_grad_norm: float | None = None

        for step_idx, batch in progress_bar:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps

            loss.backward()

            if step_idx % args.gradient_accumulation_steps == 0:
                grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
                latest_grad_norm = float(grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * args.gradient_accumulation_steps
            # if step_idx % 10 == 0:
            metrics = {
                "train/loss": running_loss / step_idx,
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
                "train/current_epoch": epoch + 1,
                "global_step": (epoch * len(train_loader)) + step_idx,
            }
            if latest_grad_norm is not None:
                metrics["train/grad_norm"] = latest_grad_norm
            wandb.log(metrics)
            progress_bar.set_postfix({"loss": running_loss / step_idx})

        epoch_loss = running_loss / len(train_loader)
        wandb.log({
            "train/epoch_loss": epoch_loss,
            "epoch": epoch + 1,
        })

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
        eval_loss = eval_loss / max(len(eval_loader), 1)
        wandb.log({
            "eval/loss": eval_loss,
            "epoch": epoch + 1,
            "eval/current_epoch": epoch + 1,
        })

        save_path = output_dir / f"epoch-{epoch + 1}"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    print(f"Args: {args}")
    run_training(args)
