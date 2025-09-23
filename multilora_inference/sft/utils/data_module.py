from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset

from local_dataset import load_dataset


@dataclass
class DataConfig:
    dataset_name: str
    max_samples: int = 10_000
    test_size: float = 0.1
    max_length: int | None = None
    batch_size: int = 4
    seed: int = 42
    num_workers: int = 2


class ChatTemplateDataset(Dataset):
    """Torch dataset applying the chat template and label masking."""

    def __init__(self, hf_dataset: HFDataset, tokenizer, max_length: int, seed: int = 42) -> None:
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def _build_messages(self, instruction: str, output: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        prompt_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": output},
        ]
        return prompt_messages, full_messages

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.dataset[idx]
        instruction = row["instruction"]
        output = row["output"]

        prompt_messages, full_messages = self._build_messages(instruction, output)

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
        )

        # Defer tokenization and padding to collate for fixed-length batching
        sample = {
            "prompt_text": prompt_text,
            "full_text": full_text,
        }
        return sample


def build_dataloaders(tokenizer, config: DataConfig) -> Tuple[DataLoader, DataLoader, dict[str, int]]:
    """Create train and eval dataloaders with chat templated supervision."""
    # Exclude test set from training
    raw_df, _ = load_dataset(config.dataset_name)
    sampled_df = raw_df.sample(
        n=min(len(raw_df), config.max_samples),
        random_state=config.seed,
        replace=False,
    ).reset_index(drop=True)

    hf_dataset = HFDataset.from_pandas(sampled_df, preserve_index=False)
    split_dataset = hf_dataset.train_test_split(test_size=config.test_size, seed=config.seed)

    tokenizer_max_length = config.max_length or getattr(tokenizer, "model_max_length", 4096)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    train_dataset = ChatTemplateDataset(split_dataset["train"], tokenizer, tokenizer_max_length, config.seed)
    eval_dataset = ChatTemplateDataset(split_dataset["test"], tokenizer, tokenizer_max_length, config.seed)

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Tokenize with dynamic in-batch padding up to longest sequence; truncate to max_length
        prompt_texts = [item["prompt_text"] for item in batch]
        full_texts = [item["full_text"] for item in batch]

        full_tok = tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=tokenizer_max_length,
            return_tensors="pt",
        )
        prompt_tok = tokenizer(
            prompt_texts,
            padding=False,
            truncation=True,
            max_length=tokenizer_max_length,
            return_tensors=None,
        )

        input_ids = full_tok["input_ids"]
        attention_mask = full_tok["attention_mask"]
        labels = input_ids.clone()

        # # Ignore loss on padding positions
        # labels[attention_mask == 0] = -100

        # Mask out prompt tokens in labels to avoid training on inputs
        for i in range(len(batch)):
            prompt_len = len(prompt_tok["input_ids"][i])
            mask_len = min(prompt_len, labels.size(1))
            labels[i, :mask_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    metadata = {
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "max_length": tokenizer_max_length,
        "pad_token_id": pad_token_id,
    }

    return train_loader, eval_loader, metadata
