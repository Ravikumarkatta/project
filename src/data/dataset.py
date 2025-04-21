# src/data/dataset.py
import json
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


class BibleDataset(Dataset):
    """Dataset for Bible verses."""

    def __init__(
        self,
        bible_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        sample_ratio: float = 1.0,
    ):
        """
        Initialize dataset from Bible data.

        Args:
            bible_path: Path to Bible JSON file.
            tokenizer: HuggingFace tokenizer to use.
            max_length: Maximum sequence length.
            sample_ratio: Ratio of verses to sample (0.0-1.0).
        """
        self.data = self._load_and_process_data(bible_path, sample_ratio)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_and_process_data(self, bible_path: str, sample_ratio: float) -> List[str]:
        """Load Bible data from JSON file and convert to flat list of verses."""
        with open(bible_path, "r", encoding="utf-8") as f:
            bible_data = json.load(f)

        verses = []
        for book, chapters in bible_data.items():
            for chapter, verse_dict in chapters.items():
                for verse_num, verse_text in verse_dict.items():
                    formatted_verse = f"{book} {chapter}:{verse_num} - {verse_text}"
                    verses.append(formatted_verse)

        # Sample if needed
        if sample_ratio < 1.0:
            random.shuffle(verses)
            verses = verses[:int(len(verses) * sample_ratio)]

        return verses

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized verse."""
        verse = self.data[idx]

        # Tokenize
        tokenized = self.tokenizer(
            verse,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
        }


class BibleInstructionDataset(Dataset):
    """Dataset for instruction fine-tuning with biblical data."""

    def __init__(
        self, 
        data_path: str, 
        tokenizer: PreTrainedTokenizer, 
        max_length: int = 512,
        instruction_types: Optional[List[str]] = None
    ):
        """
        Initialize dataset from instruction data.

        Args:
            data_path: Path to instruction JSON file.
            tokenizer: HuggingFace tokenizer to use.
            max_length: Maximum sequence length.
            instruction_types: If provided, filter by these instruction types.
        """
        self.data = self._load_and_filter_data(data_path, instruction_types)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_and_filter_data(
        self, data_path: str, instruction_types: Optional[List[str]]
    ) -> List[Dict]:
        """Load instruction data from JSON file and filter if needed."""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if instruction_types:
            return [item for item in data if item.get("instruction_type") in instruction_types]
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized instruction example."""
        item = self.data[idx]

        # Format as instruction prompt
        instruction = item["instruction"]
        input_text = item["input"]
        output = item["output"]

        # Format prompt according to instruction tuning format
        prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: "

        # Tokenize prompt
        prompt_tokenized = self.tokenizer(
            prompt,
            max_length=self.max_length // 2,  # Reserve half length for output
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize output (labels)
        output_tokenized = self.tokenizer(
            output,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Combine input_ids: prompt followed by output
        input_ids = torch.cat(
            [
                prompt_tokenized["input_ids"].squeeze(),
                output_tokenized["input_ids"].squeeze(),
            ]
        )[: self.max_length]

        # Create attention mask (1 for prompt and output tokens, 0 for padding)
        attention_mask = torch.cat(
            [
                prompt_tokenized["attention_mask"].squeeze(),
                output_tokenized["attention_mask"].squeeze(),
            ]
        )[: self.max_length]

        # Create labels tensor: -100 for prompt tokens (ignored in loss), actual ids for output
        labels = torch.cat(
            [
                torch.full_like(prompt_tokenized["input_ids"].squeeze(), -100),
                output_tokenized["input_ids"].squeeze(),
            ]
        )[: self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# Also define BiblicalDataset for backward compatibility
class BiblicalDataset(Dataset):
    """Custom Dataset for biblical data (for backward compatibility)."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Initialize the dataset with input_ids, labels, and attention_mask.

        Args:
            input_ids: Tensor of input token IDs.
            labels: Tensor of label token IDs.
            attention_mask: Tensor of attention masks.
        """
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
        }


def create_bible_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for Bible training and validation.
    
    Args:
        train_path: Path to training Bible data.
        val_path: Path to validation Bible data.
        tokenizer: HuggingFace tokenizer to use.
        batch_size: Batch size for DataLoaders.
        max_length: Maximum sequence length.
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = BibleDataset(train_path, tokenizer, max_length)
    val_dataset = BibleDataset(val_path, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_instruction_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    instruction_types: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for instruction fine-tuning.
    
    Args:
        train_path: Path to training instruction data.
        val_path: Path to validation instruction data.
        tokenizer: HuggingFace tokenizer to use.
        batch_size: Batch size for DataLoaders.
        max_length: Maximum sequence length.
        instruction_types: Optional list of instruction types to filter.
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = BibleInstructionDataset(
        train_path, tokenizer, max_length, instruction_types
    )
    val_dataset = BibleInstructionDataset(
        val_path, tokenizer, max_length, instruction_types
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
