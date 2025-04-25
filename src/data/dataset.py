import json
import os
import random
import logging
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BibleDataset(Dataset):
    """Dataset for Bible verses with improved error handling and tokenization."""

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

        Raises:
            FileNotFoundError: If bible_path doesn't exist.
            ValueError: If data is invalid or empty.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_and_process_data(bible_path, sample_ratio)

    def _load_and_process_data(self, bible_path: str, sample_ratio: float) -> List[str]:
        """Load and validate Bible data from JSON file."""
        if not os.path.exists(bible_path):
            raise FileNotFoundError(f"Bible data file not found: {bible_path}")

        try:
            with open(bible_path, "r", encoding="utf-8") as f:
                bible_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in bible data file: {e}")

        verses = []
        for book, chapters in bible_data.items():
            if not isinstance(chapters, dict):
                logger.warning(f"Skipping invalid chapters for book {book}")
                continue
                
            for chapter, verse_dict in chapters.items():
                if not isinstance(verse_dict, dict):
                    logger.warning(f"Skipping invalid verses for {book} {chapter}")
                    continue
                    
                for verse_num, verse_text in verse_dict.items():
                    if not verse_text.strip():
                        continue
                    formatted_verse = f"{book} {chapter}:{verse_num} - {verse_text}"
                    verses.append(formatted_verse)

        if not verses:
            raise ValueError("No valid verses found in bible data")

        if sample_ratio < 1.0:
            random.shuffle(verses)
            verses = verses[:int(len(verses) * sample_ratio)]

        logger.info(f"Loaded {len(verses)} verses")
        return verses

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized verse with proper error handling."""
        verse = self.data[idx]

        try:
            tokenized = self.tokenizer(
                verse,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
            }
        except Exception as e:
            logger.error(f"Error tokenizing verse at index {idx}: {verse}")
            raise RuntimeError(f"Tokenization failed: {e}") from e


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

        Raises:
            FileNotFoundError: If data_path doesn't exist.
            ValueError: If data is invalid or empty.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_and_filter_data(data_path, instruction_types)

    def _load_and_filter_data(
        self, data_path: str, instruction_types: Optional[List[str]]
    ) -> List[Dict]:
        """Load and validate instruction data from JSON file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Instruction data file not found: {data_path}")

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in instruction data file: {e}")

        if not isinstance(data, list):
            raise ValueError("Instruction data should be a list of examples")

        if instruction_types:
            data = [item for item in data if item.get("instruction_type") in instruction_types]

        # Validate required fields
        valid_data = []
        for item in data:
            if not all(key in item for key in ["instruction", "input", "output"]):
                logger.warning(f"Skipping invalid item missing required fields: {item}")
                continue
            valid_data.append(item)

        if not valid_data:
            raise ValueError("No valid instruction examples found")

        logger.info(f"Loaded {len(valid_data)} instruction examples")
        return valid_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized instruction example with proper labels."""
        item = self.data[idx]

        try:
            # Format full text including instruction, input and output
            text = (
                f"Instruction: {item['instruction']}\n\n"
                f"Input: {item['input']}\n\n"
                f"Output: {item['output']}"
            )
            
            # Tokenize all at once
            tokenized = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Create labels (-100 for prompt part, actual tokens for output)
            labels = tokenized["input_ids"].clone()
            output_start = text.find("Output: ") + len("Output: ")
            
            # Convert everything before output to -100 (ignore in loss)
            labels[0, :output_start] = -100
            
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0),
            }
        except Exception as e:
            logger.error(f"Error tokenizing instruction at index {idx}: {item}")
            raise RuntimeError(f"Tokenization failed: {e}") from e


def create_bible_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for Bible training and validation with proper collation.

    Args:
        train_path: Path to training Bible data.
        val_path: Path to validation Bible data.
        tokenizer: HuggingFace tokenizer to use.
        batch_size: Batch size for DataLoaders.
        max_length: Maximum sequence length.
        num_workers: Number of workers for data loading.

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        RuntimeError: If dataloader creation fails.
    """
    try:
        train_dataset = BibleDataset(train_path, tokenizer, max_length)
        val_dataset = BibleDataset(val_path, tokenizer, max_length)
        
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            }
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(
            f"Created dataloaders with {len(train_dataset)} train and "
            f"{len(val_dataset)} val examples"
        )
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise RuntimeError(f"Dataloader creation failed: {e}") from e


def create_instruction_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    instruction_types: Optional[List[str]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for instruction fine-tuning with proper collation.

    Args:
        train_path: Path to training instruction data.
        val_path: Path to validation instruction data.
        tokenizer: HuggingFace tokenizer to use.
        batch_size: Batch size for DataLoaders.
        max_length: Maximum sequence length.
        instruction_types: Optional list of instruction types to filter.
        num_workers: Number of workers for data loading.

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        RuntimeError: If dataloader creation fails.
    """
    try:
        train_dataset = BibleInstructionDataset(
            train_path, tokenizer, max_length, instruction_types
        )
        val_dataset = BibleInstructionDataset(
            val_path, tokenizer, max_length, instruction_types
        )
        
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
                'labels': torch.stack([x['labels'] for x in batch]),
            }
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(
            f"Created instruction dataloaders with {len(train_dataset)} train and "
            f"{len(val_dataset)} val examples"
        )
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Failed to create instruction dataloaders: {e}")
        raise RuntimeError(f"Instruction dataloader creation failed: {e}") from e
