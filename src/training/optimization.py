from typing import Iterator, Tuple

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """
    Create a schedule with linear learning rate warmup and decay.

    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: The index of last epoch

    Returns:
        Learning rate scheduler
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_optimizer_and_scheduler(
    params: Iterator[torch.nn.Parameter],
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
) -> Tuple[Optimizer, LambdaLR]:
    """
    Create optimizer and scheduler for training.

    Args:
        params: Model parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay factor
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        epsilon: Adam epsilon parameter
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create parameter groups for different weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Initialize AdamW optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=lr, betas=(beta1, beta2), eps=epsilon
    )

    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    return optimizer, scheduler
