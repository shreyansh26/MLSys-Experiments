from train_utils import (
    load_instruction_dataset,
    compute_metrics,
    postprocess_text,
    preprocess_function,
)
from transformers import AutoTokenizer
from tokenizers import AddedToken
from datasets import load_dataset, Features, Value, ClassLabel, Sequence
from random import randrange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import concatenate_datasets
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

def get_datasets(train_file_path, valid_file_path, tokenizer, MODEL_NAME, MAX_TOKEN_COUNT):
    dataset = load_instruction_dataset(
            train_path=train_file_path, valid_path=valid_file_path, max_token_count=MAX_TOKEN_COUNT
        )
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['valid'])}")

    print(dataset["train"][0])

    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["valid"]]).map(
        lambda x: tokenizer(
            x["prompt"] + [" "] * (len(x)) + x["input_text"], truncation=True
        ),
        batched=True,
        remove_columns=["input_text", "output_text", "prompt"],
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["valid"]]).map(
        lambda x: tokenizer(x["output_text"], truncation=True),
        batched=True,
        remove_columns=["input_text", "output_text", "prompt"],
    )
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    tokenized_dataset = dataset.map(
        preprocess_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
        },
        batched=True,
        remove_columns=["prompt", "input_text", "output_text"],
    )
    tokenized_dataset = tokenized_dataset.shuffle(seed=100)
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
    return tokenized_dataset