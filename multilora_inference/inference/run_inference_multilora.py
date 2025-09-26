import sys
import os
import time
import argparse
from pathlib import Path
from typing import Any, List
import random

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sft.local_dataset import load_dataset
from analyze_lora_adapters import get_A_B_weights, get_base_model_config, get_multilora_A_B_weights
from modeling_llama_multilora import LlamaForCausalLM


DEFAULT_OUTPUT_ROOT = "/mnt/ssd2/shreyansh/models/multilora"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_BATCH_SIZE = 4

LORA_MAPPING = {
    "ifeval_like_data": 0,
    "multilingual_cohere_aya": 1,
    "opc_evol_instruct": 2,
    "text_to_sql": 3,
    "infinity_instruct": 4,
    "numina_math": 5,
    "opc_sft_educational": 6,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with LoRA-adapted model")
    parser.add_argument(
        "--lora-inference-mode",
        type=str,
        default="gbmm",
        choices=["bgmv_cuda", "bgmv_triton", "gbmm"],
        help="Multi-LoRA inference mode",
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
        default=None,
        help="Name of dataset used both to craft inputs and optionally to locate checkpoints",
    )
    parser.add_argument(
        "--dataset-names",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset names to sample randomly from for a batch",
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


def get_lora_checkpoint_dir() -> List[Path]:
    return [
        Path(DEFAULT_OUTPUT_ROOT) / name / f"epoch-2"
        for name, epoch in LORA_MAPPING.items()
    ]


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


def get_instructions_outputs_from_datasets(dataset_names: list[str], batch_size: int) -> tuple[list[str], list[str], list[int]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if not dataset_names:
        raise ValueError("Provide at least one dataset name")

    # Preload all datasets and validate columns
    dataframes = []
    for name in dataset_names:
        _, df = load_dataset(name)
        if len(df) == 0:
            raise ValueError(f"Dataset '{name}' is empty")
        if "instruction" not in df.columns:
            raise KeyError(f"Dataset '{name}' does not contain 'instruction' column")
        if "output" not in df.columns:
            raise KeyError(f"Dataset '{name}' does not contain 'output' column")
        dataframes.append(df)

    instructions: list[str] = []
    outputs: list[str] = []
    dataset_indices: list[int] = []

    for _ in range(batch_size):
        ds_idx = random.randrange(len(dataset_names))
        df = dataframes[ds_idx]
        row_idx = random.randrange(len(df))
        row = df.iloc[row_idx]
        instructions.append(str(row["instruction"]))
        outputs.append(str(row["output"]))
        dataset_indices.append(ds_idx)

    return instructions, outputs, dataset_indices


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    ckpt_dirs = get_lora_checkpoint_dir()
    if not all(ckpt_dir.exists() for ckpt_dir in ckpt_dirs):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dirs}")

    base_model_config, base_model_name, adapter_config = get_base_model_config(ckpt_dirs[0])
    lora_A_weights, lora_B_weights = get_multilora_A_B_weights(ckpt_dirs, base_model_config, mode=args.lora_inference_mode)

    print(lora_A_weights["q_proj"].shape)

    model = LlamaForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.model.lora_A_weights = lora_A_weights
    model.model.lora_B_weights = lora_B_weights
    model.model.lora_scale = adapter_config["lora_alpha"] / adapter_config["r"]
    
    model = model.to(device)
    model.eval()

    tokenizer = load_tokenizer(ckpt_dirs[0], args.model_name)

    dataset_names = list(LORA_MAPPING.keys())
    if dataset_names is not None and len(dataset_names) > 0:
        instructions, reference_outputs, dataset_indices = get_instructions_outputs_from_datasets(
            dataset_names, DEFAULT_BATCH_SIZE
        )
        lora_indices = dataset_indices
        print(lora_indices)
    else:
        instructions, reference_outputs, sample_indices = get_instructions_outputs_from_dataset(
            args.dataset_name, args.sample_index, DEFAULT_BATCH_SIZE
        )
        lora_indices = [LORA_MAPPING[args.dataset_name]] * DEFAULT_BATCH_SIZE
        print(lora_indices)
    
    lora_indices = torch.tensor(lora_indices, device=device)
    inputs = build_chat_inputs(tokenizer, instructions, device)
    print("-" * 100)

    # Warmup: run a minimal generation to compile kernels before timing
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            lora_indices=lora_indices,
            lora_inference_mode=args.lora_inference_mode
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
            lora_indices=lora_indices,
            lora_inference_mode=args.lora_inference_mode
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
        zip(lora_indices, instructions, reference_outputs, generated_texts)
    ):
        print(f"Sample {batch_idx} (lora/dataset index {dataset_idx}):")
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
    # CUDA_VISIBLE_DEVICES=0 python run_inference_multilora.py --lora-inference-mode bgmv_cuda
    # CUDA_VISIBLE_DEVICES=0 python run_inference_multilora.py --lora-inference-mode bgmv_triton
    main()
