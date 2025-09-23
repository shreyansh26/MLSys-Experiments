import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from transformers import get_cosine_schedule_with_warmup


def set_random_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parameter_weight_decay_filter(named_params: Iterable[tuple[str, nn.Parameter]]) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return decay_params, no_decay_params


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: Sequence[float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer targeting only trainable parameters."""
    decay_params, no_decay_params = _parameter_weight_decay_filter(model.named_parameters())

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    if not param_groups:
        raise ValueError("No trainable parameters found to optimize.")

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, eps=eps)
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a cosine scheduler with warmup compatible with HF get_cosine_schedule_with_warmup."""
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler
