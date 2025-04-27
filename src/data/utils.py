"""
Utility functions for data processing in biblical text analysis.
"""

from typing import Dict, List

import torch


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched tensors
    """
    # Initialize batch dictionary
    batched = {"input_ids": [], "attention_mask": [], "labels": []}

    # Gather each field from all samples
    for sample in batch:
        for key in batched:
            batched[key].append(sample[key])

    # Stack all tensors
    for key in batched:
        batched[key] = torch.stack(batched[key])

    return batched


def pad_sequences(sequences: List[List[int]], padding_value: int = 0) -> torch.Tensor:
    """
    Pad sequences to the same length.

    Args:
        sequences: List of token ID sequences
        padding_value: Value to use for padding

    Returns:
        Padded tensor of shape (batch_size, max_length)
    """
    # Find max length in batch
    max_len = max(len(seq) for seq in sequences)

    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        padded = seq + [padding_value] * (max_len - len(seq))
        padded_sequences.append(padded)

    return torch.tensor(padded_sequences)


def create_attention_mask(
    input_ids: torch.Tensor, padding_value: int = 0
) -> torch.Tensor:
    """
    Create attention mask from input IDs.

    Args:
        input_ids: Tensor of token IDs
        padding_value: Value used for padding

    Returns:
        Attention mask tensor (1 for tokens, 0 for padding)
    """
    return (input_ids != padding_value).float()
