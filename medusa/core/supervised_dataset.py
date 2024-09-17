from dataclasses import dataclass
import os
from typing import Dict, Sequence
from torch.utils.data import Dataset
import datasets
import logging
import torch.distributed as dist
import torch
import transformers
import copy
import math


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|unk|>"


def fmt_prompt(prompt):
    return f"### Instructions:\n{prompt}\n\n### Response:"


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    # print(f'in _tokenize: {len(tokenized_list)}, {tokenized_list[0].input_ids.shape}')

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    train_on_inputs: bool,
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources = [f"{fmt_prompt(question)}" for question in samples["instruction"]]
    targets = [f"{answer}{tokenizer.eos_token}" for answer in samples["output"]]
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        if not train_on_inputs:
            label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

def preprocess_instruct(
    train_on_inputs: bool,
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources = [question for question in samples["instruction"]]
    targets = [f"{answer}{tokenizer.eos_token}" for answer in samples["output"]]
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        if not train_on_inputs:
            label[:source_len] = IGNORE_INDEX

    # print(input_ids)
    # print(labels)
    # print(f"In preprocess: input_ids.shape {len(input_ids)}, {input_ids[0].shape}")
    # print(f"In preprocess: labels.shape  {len(labels)}, {labels[0].shape}")

    return dict(input_ids=input_ids, labels=labels)

def _filter_tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    samples = []
    for text in strings:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        if tokens.input_ids.squeeze().numel() < tokenizer.model_max_length:
            samples.append(True)
        else:
            samples.append(False)

    return samples


def filter_long_samples(
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    sources = [f"{fmt_prompt(question)}" for question in samples["instruction"]]
    targets = [f"{answer}{tokenizer.eos_token}" for answer in samples["output"]]
    examples = [s + t for s, t in zip(sources, targets)]

    return _filter_tokenize_fn(examples, tokenizer)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        train_on_inputs: bool,
        tokenizer: transformers.PreTrainedTokenizer,
        data_paths,
        use_instruct_format=False,
        limit=None,
        workers=None
    ):
        super(SupervisedDataset, self).__init__()
        if workers is None:
            workers = math.ceil(os.cpu_count() / dist.get_world_size())
        logging.warning(f"TOKENIZING WITH NUM_WORKERS: {workers}")
        preprocess_fn = preprocess_instruct if use_instruct_format else preprocess
        dataset = (
            datasets.load_dataset(
                "json",
                data_files=data_paths,
                split=f"train[0:{limit}]" if limit else "train",
            )
            .filter(
                lambda samples: filter_long_samples(samples, tokenizer),
                batched=True,
                batch_size=3000,
                num_proc=workers,
            )
            .map(
                lambda samples: preprocess_fn(train_on_inputs, samples, tokenizer),
                batched=True,
                batch_size=3000,
                num_proc=workers,
            )
        )

        self.input_ids = dataset["input_ids"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=torch.tensor(self.input_ids[i]),
            labels=torch.tensor(self.labels[i]),
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(f"In datacollator: input_ids before : {len(instances)}, {instances[0]['input_ids'].shape}")
        # print(f"In datacollator: labels before : {len(instances)}, {instances[0]['labels'].shape}")
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # print(f"In datacollator: after input_ids.shape {input_ids.shape}")
        # print(f"In datacollator: after labels.shape {labels.shape}")
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
