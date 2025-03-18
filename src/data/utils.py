# src/data/utils.py
from typing import List, Dict, Tuple
import torch

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to convert dataset outputs into (input_ids, target_ids) for training.
    
    Args:
        batch: List of dictionary items from dataset.
        
    Returns:
        Tuple of (input_ids, target_ids) as tensors.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    # For language modeling, target_ids are the same as input_ids (shifted in the model)
    target_ids = input_ids.clone()
    return input_ids, target_ids
