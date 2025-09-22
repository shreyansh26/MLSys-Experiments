from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from local_dataset import load_dataset


DEFAULT_OUTPUT_ROOT = "/mnt/ssd2/shreyansh/models/multilora"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with LoRA-adapted model")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory containing adapter and tokenizer (e.g. .../dataset/epoch-2)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of dataset used both to craft inputs and optionally to locate checkpoints",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to load for dataset-based lookup",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base model name (used for tokenizer fallback if needed)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Load model in bfloat16 if available",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Row index from dataset to build the input instruction",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Force CUDA if available",
    )
    return parser.parse_args()


def resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        return Path(args.checkpoint_dir)
    if args.dataset_name and args.epoch:
        return Path(DEFAULT_OUTPUT_ROOT) / args.dataset_name / f"epoch-{args.epoch}"
    raise ValueError("Provide either --checkpoint-dir or both --dataset-name and --epoch")


def load_tokenizer(checkpoint_path: Path, fallback_model_name: str) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    return tokenizer


def build_chat_inputs(tokenizer: Any, instruction: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in model_inputs.items()}


def get_instruction_from_dataset(dataset_name: str, sample_index: int) -> str:
    df = load_dataset(dataset_name)
    if len(df) == 0:
        raise ValueError(f"Dataset '{dataset_name}' is empty")
    # Allow out-of-range indices by wrapping around
    safe_index = sample_index % len(df)
    row = df.iloc[safe_index]
    if "instruction" not in row:
        raise KeyError("Dataset row does not contain 'instruction' column")
    return str(row["instruction"]) if row["instruction"] is not None else ""


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    ckpt_dir = resolve_checkpoint_dir(args)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    # AutoPeftModelForCausalLM will restore the base model plus attached LoRA adapters
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ckpt_dir),
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model = model.to(device)
    model.eval()

    tokenizer = load_tokenizer(ckpt_dir, args.model_name)

    instruction = get_instruction_from_dataset(args.dataset_name, args.sample_index)
    inputs = build_chat_inputs(tokenizer, instruction, device)

    print(instruction)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove the prompt portion for clean decoding
    generated_only_ids = generated_ids[:, inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(generated_only_ids[0], skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    # Respect external CUDA_VISIBLE_DEVICES; user can set to target a specific GPU
    # Example: CUDA_VISIBLE_DEVICES=4 python run_inference.py --dataset-name <name> --epoch 2 --sample-index 0
    main()


