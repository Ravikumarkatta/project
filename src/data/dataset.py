"""Dataset classes for biblical text data processing."""

import json
import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenization import BiblicalTokenizer

logger = logging.getLogger(__name__)


class BibleDataset(Dataset):
    """Dataset for training on biblical content."""

    def __init__(
        self,
        data_file: str,
        tokenizer: BiblicalTokenizer,
        max_length: int = 512,
        sample_ratio: float = 1.0,
    ):
        """
        Initialize the dataset.

        Args:
            data_file: Path to JSON file containing Bible data
            tokenizer: BiblicalTokenizer instance
            max_length: Maximum sequence length
            sample_ratio: Ratio of data to use (for debugging/testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load Bible data
        with open(data_file, "r", encoding="utf-8") as f:
            self.bible_data = json.load(f)

        # Convert hierarchical Bible data into flat list of verses
        self.verses = []
        for book, chapters in self.bible_data.items():
            for chapter, verses in chapters.items():
                for verse_num, text in verses.items():
                    self.verses.append(
                        {"reference": f"{book} {chapter}:{verse_num}", "text": text}
                    )

        # Sample data if needed
        if sample_ratio < 1.0:
            num_samples = int(len(self.verses) * sample_ratio)
            self.verses = self.verses[:num_samples]

        logger.info(f"Loaded {len(self.verses)} verses from {data_file}")

    def __len__(self) -> int:
        return len(self.verses)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized verse data."""
        verse = self.verses[idx]

        # Combine reference and text
        full_text = f"{verse['reference']} {verse['text']}"

        # Tokenize
        encoding = self.tokenizer.tokenize(
            full_text, return_tensors=None  # We'll convert to tensors here
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
        }


class BibleInstructionDataset(Dataset):
    """Dataset for instruction fine-tuning with biblical data."""

    def __init__(
        self,
        data_file: str,
        tokenizer: BiblicalTokenizer,
        max_length: int = 512,
        instruction_types: Optional[List[str]] = None,
    ):
        """
        Initialize the instruction dataset.

        Args:
            data_file: Path to JSON file containing instruction data
            tokenizer: BiblicalTokenizer instance
            max_length: Maximum sequence length
            instruction_types: List of instruction types to include
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load instruction data
        with open(data_file, "r", encoding="utf-8") as f:
            self.instruction_data = json.load(f)

        # Filter by instruction types if specified
        if instruction_types:
            self.instruction_data = [
                item
                for item in self.instruction_data
                if item["instruction_type"] in instruction_types
            ]

        logger.info(f"Loaded {len(self.instruction_data)} instruction examples")

    def __len__(self) -> int:
        return len(self.instruction_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized instruction data."""
        item = self.instruction_data[idx]

        # Tokenize instruction, input, and output
        encoding = self.tokenizer.tokenize_instruction_data(
            instruction=item["instruction"],
            input_text=item["input"],
            output=item["output"],
            return_tensors=None,
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "labels": torch.tensor(encoding["labels"], dtype=torch.long),
        }


def create_bible_dataloaders(
    train_data: str,
    val_data: str,
    tokenizer: BiblicalTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        train_data: Path to training data file
        val_data: Path to validation data file
        tokenizer: BiblicalTokenizer instance
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = BibleDataset(train_data, tokenizer, max_length=max_length)
    val_dataset = BibleDataset(val_data, tokenizer, max_length=max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_instruction_dataloaders(
    train_data: str,
    val_data: str,
    tokenizer: BiblicalTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    instruction_types: Optional[List[str]] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for instruction fine-tuning.

    Args:
        train_data: Path to training instruction data file
        val_data: Path to validation instruction data file
        tokenizer: BiblicalTokenizer instance
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading
        instruction_types: Optional list of instruction types to include

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = BibleInstructionDataset(
        train_data,
        tokenizer,
        max_length=max_length,
        instruction_types=instruction_types,
    )
    val_dataset = BibleInstructionDataset(
        val_data, tokenizer, max_length=max_length, instruction_types=instruction_types
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
