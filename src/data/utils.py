# src/data/utils.py
import torch
from typing import Dict, List


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader to handle variable length sequences.
    
    Args:
        batch: List of dictionaries with tensors
        
    Returns:
        Dictionary with batched tensors
    """
    batch_dict = {}
    
    # Get all keys from the first item
    keys = batch[0].keys()
    
    for key in keys:
        # Stack tensors for each key
        batch_dict[key] = torch.stack([item[key] for item in batch])
    
    return batch_dict
