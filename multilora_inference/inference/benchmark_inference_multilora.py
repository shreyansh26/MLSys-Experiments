import sys
import os
import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from run_inference_multilora import (
    prepare_batch_data,
    load_multilora_model,
    load_tokenizer,
    run_inference,
    get_lora_checkpoint_dir,
    LORA_MAPPING,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multi-LoRA inference across all modes")
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["gbmm", "bgmv_triton", "bgmv_cuda", "sgmv_triton"],
        choices=["gbmm", "bgmv_triton", "bgmv_cuda", "sgmv_triton"],
        help="List of inference modes to benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Get checkpoint directories
    ckpt_dirs = get_lora_checkpoint_dir()

    # Prepare batch data ONCE (same for all modes)
    dataset_names = list(LORA_MAPPING.keys())
    instructions, reference_outputs, lora_indices = prepare_batch_data(
        dataset_names, args.batch_size, add_no_lora_sample=True
    )
    
    lora_indices_tensor = torch.tensor(lora_indices, device=device)
    
    print("=" * 100)
    print("BENCHMARK CONFIGURATION")
    print("=" * 100)
    print(f"Number of LoRA adapters: {len(LORA_MAPPING)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batch LoRA indices: {lora_indices_tensor.tolist()}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Modes to benchmark: {args.modes}")
    print("=" * 100)
    print()

    # Store results for comparison
    results = {}

    # Run inference for each mode
    for mode in args.modes:
        print("=" * 100)
        print(f"BENCHMARKING MODE: {mode}")
        print("=" * 100)
        
        # Load model with this mode's LoRA weights
        model, metadata = load_multilora_model(ckpt_dirs, torch_dtype, device, mode)
        tokenizer = load_tokenizer(ckpt_dirs[0], args.model_name)
        
        # Run inference
        generated_texts, tokens_per_second, total_generated_tokens = run_inference(
            model=model,
            tokenizer=tokenizer,
            instructions=instructions,
            lora_indices=lora_indices_tensor,
            lora_inference_mode=mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            warmup=True,
        )
        
        generation_seconds = total_generated_tokens / tokens_per_second if tokens_per_second else 0.0
        
        results[mode] = {
            "tokens_per_second": tokens_per_second,
            "total_tokens": total_generated_tokens,
            "generation_seconds": generation_seconds,
            "generated_texts": generated_texts,
        }
        
        print(f"\n{mode.upper()} Results:")
        print(f"  Throughput: {tokens_per_second:.2f} tokens/sec")
        print(f"  Total tokens: {total_generated_tokens}")
        print(f"  Time: {generation_seconds:.2f} seconds")
        print()
        
        # Free model memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print comparison summary
    print("=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Mode':<20} {'Throughput (tok/s)':<25} {'Time (s)':<15} {'Total Tokens':<15}")
    print("-" * 100)
    
    # Sort by throughput (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["tokens_per_second"], reverse=True)
    
    for mode, result in sorted_results:
        print(f"{mode:<20} {result['tokens_per_second']:<25.2f} {result['generation_seconds']:<15.2f} {result['total_tokens']:<15}")
    
    print("=" * 100)
    
    # Calculate speedups relative to the slowest
    if len(sorted_results) > 1:
        slowest_throughput = sorted_results[-1][1]["tokens_per_second"]
        print("\nSpeedup vs slowest mode:")
        print("-" * 100)
        for mode, result in sorted_results:
            speedup = result["tokens_per_second"] / slowest_throughput if slowest_throughput > 0 else 0.0
            print(f"{mode:<20} {speedup:.2f}x")
        print("=" * 100)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python benchmark_inference_multilora.py
    # CUDA_VISIBLE_DEVICES=0 python benchmark_inference_multilora.py --modes gbmm sgmv_triton
    # CUDA_VISIBLE_DEVICES=0 python benchmark_inference_multilora.py --max_new_tokens 256 --batch-size 32
    main()

