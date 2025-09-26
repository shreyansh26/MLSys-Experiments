import sys
import os
import time
import argparse
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sft.local_dataset import load_dataset
from analyze_lora_adapters import get_A_B_weights, get_base_model_config
from modeling_llama import LlamaForCausalLM


DEFAULT_OUTPUT_ROOT = "/mnt/ssd2/shreyansh/models/multilora"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_BATCH_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with LoRA-adapted model")
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="peft",
        choices=["base_model", "peft", "custom_lora"],
        help="Inference mode",
    )
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
    tokenizer.padding_side = "left"
    return tokenizer


def build_chat_inputs(tokenizer: Any, instructions: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    prompts = []
    for instruction in instructions:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs_on_device = {k: v.to(device) for k, v in model_inputs.items()}
    return inputs_on_device


def get_instructions_outputs_from_dataset(dataset_name: str, start_index: int, batch_size: int) -> tuple[list[str], list[str], list[int]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    # Load the test split
    _, df = load_dataset(dataset_name)
    if len(df) == 0:
        raise ValueError(f"Dataset '{dataset_name}' is empty")
    if "instruction" not in df.columns:
        raise KeyError("Dataset does not contain 'instruction' column")
    if "output" not in df.columns:
        raise KeyError("Dataset does not contain 'output' column")

    instructions: list[str] = []
    outputs: list[str] = []
    sample_indices: list[int] = []
    for offset in range(batch_size):
        safe_index = (start_index + offset) % len(df)
        row = df.iloc[safe_index]
        instructions.append(str(row["instruction"]))
        outputs.append(str(row["output"]))
        sample_indices.append(int(safe_index))

    return instructions, outputs, sample_indices


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    ckpt_dir = resolve_checkpoint_dir(args)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    if args.inference_mode == "base_model":
        _, base_model_name, _ = get_base_model_config(ckpt_dir)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=None,
        )
    elif args.inference_mode == "peft":
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            torch_dtype=torch_dtype,
            device_map=None,
        )
    elif args.inference_mode == "custom_lora":
        base_model_config, base_model_name, adapter_config = get_base_model_config(ckpt_dir)
        lora_A_weights, lora_B_weights = get_A_B_weights(ckpt_dir, base_model_config)

        model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        model.model.lora_A_weights = lora_A_weights
        model.model.lora_B_weights = lora_B_weights
        model.model.lora_scale = adapter_config["lora_alpha"] / adapter_config["r"]
    else:
        raise ValueError(f"Invalid inference mode: {args.inference_mode}")
    
    model = model.to(device)
    model.eval()

    tokenizer = load_tokenizer(ckpt_dir, args.model_name)

    instructions, reference_outputs, sample_indices = get_instructions_outputs_from_dataset(
        args.dataset_name, args.sample_index, DEFAULT_BATCH_SIZE
    )
    inputs = build_chat_inputs(tokenizer, instructions, device)

    print(
        f"Running inference on batch starting at dataset index {args.sample_index} with batch size {DEFAULT_BATCH_SIZE}."
    )
    print(f"Resolved dataset indices: {sample_indices}")
    print("-" * 100)

    # Warmup: run a minimal generation
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    input_sequence_length = int(inputs["input_ids"].shape[-1])
    generated_ids = generated_ids.cpu()
    generated_token_tensors = [seq[input_sequence_length:] for seq in generated_ids]

    total_generated_tokens = sum(tensor.numel() for tensor in generated_token_tensors)
    generation_seconds = max(end_time - start_time, 1e-8)
    tokens_per_second = total_generated_tokens / generation_seconds if total_generated_tokens else 0.0

    generated_texts = tokenizer.batch_decode(
        [tensor.tolist() for tensor in generated_token_tensors], skip_special_tokens=True
    )

    for batch_idx, (dataset_idx, instruction, reference, prediction) in enumerate(
        zip(sample_indices, instructions, reference_outputs, generated_texts)
    ):
        print(f"Sample {batch_idx} (dataset index {dataset_idx}):")
        print("Instruction:")
        print(instruction)
        print("Reference Output:")
        print(reference)
        print("Generated Output:")
        print(prediction)
        print("-" * 100)

    print(
        f"Decoding throughput: {tokens_per_second:.2f} tokens/sec "
        f"({total_generated_tokens} tokens in {generation_seconds:.2f} seconds)."
    )


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python run_inference.py --checkpoint-dir "/mnt/ssd2/shreyansh/models/multilora/text_to_sql/epoch-2" --dataset-name text_to_sql --inference-mode peft
    # CUDA_VISIBLE_DEVICES=0 python run_inference.py --checkpoint-dir "/mnt/ssd2/shreyansh/models/multilora/numina_math/epoch-2" --dataset-name text_to_sql --inference-mode custom_lora
    # CUDA_VISIBLE_DEVICES=0 python run_inference.py --dataset-name <name> --epoch 2 --sample-index 0
    main()