import json
import os
import random
import logging
from typing import Dict, List, Tuple, Optional, Any # Import Any for Dict return type

import torch
from torch.utils.data import DataLoader, Dataset
# We expect a BibleTokenizer, which wraps a PreTrainedTokenizer, not just any PreTrainedTokenizer
# from transformers import PreTrainedTokenizer # Keep this import if you still want the type hint, but the logic uses BibleTokenizer methods
from src.data.tokenization import BibleTokenizer # Import the specific tokenizer class


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BibleDataset(Dataset):
    """Dataset for Bible verses with improved error handling and tokenization."""

    def __init__(
        self,
        bible_path: str,
        tokenizer: BibleTokenizer, # Type hint should be BibleTokenizer
        max_length: int = 512, # Keep max_length in init, but tokenizer uses its own config
        sample_ratio: float = 1.0,
    ):
        """
        Initialize dataset from Bible data.

        Args:
            bible_path: Path to Bible JSON file.
            tokenizer: BibleTokenizer instance to use.
            max_length: Maximum sequence length (Note: BibleTokenizer uses its config's max_tokens).
            sample_ratio: Ratio of verses to sample (0.0-1.0).

        Raises:
            FileNotFoundError: If bible_path doesn't exist.
            ValueError: If data is invalid or empty.
        """
        self.tokenizer = tokenizer
        # self.max_length = max_length # The tokenizer handles max_length based on its config
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

                # Sort verse numbers numerically to ensure consistent order
                sorted_verse_nums = sorted(verse_dict.keys(), key=lambda x: int(x))

                for verse_num in sorted_verse_nums:
                    verse_text = verse_dict[verse_num]
                    if not verse_text or not isinstance(verse_text, str) or not verse_text.strip():
                        continue
                    # Format the verse reference and text
                    formatted_verse = f"{book} {chapter}:{verse_num} - {verse_text.strip()}"
                    verses.append(formatted_verse)

        if not verses:
            raise ValueError("No valid verses found in bible data")

        if sample_ratio < 1.0:
            random.seed(42) # Use a fixed seed for reproducibility in sampling
            random.shuffle(verses)
            verses = verses[:int(len(verses) * sample_ratio)]

        logger.info(f"Loaded {len(verses)} verses from {bible_path}")
        return verses

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized verse with proper error handling."""
        verse = self.data[idx]

        try:
            # Corrected: Call the 'tokenize' method of the BibleTokenizer instance
            # The BibleTokenizer handles max_length, truncation, padding internally
            tokenized = self.tokenizer.tokenize(
                verse,
                return_tensors="pt" # Request PyTorch tensors
            )

            # The tokenize method returns a dictionary compatible with HuggingFace outputs
            # It should contain 'input_ids' and 'attention_mask'
            if "input_ids" not in tokenized or "attention_mask" not in tokenized:
                 raise ValueError("Tokenizer did not return expected keys (input_ids, attention_mask)")

            # Ensure tensors are squeezed to remove the batch dimension added by return_tensors="pt"
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
            }
        except Exception as e:
            # Log the error and the problematic verse
            logger.error(f"Error tokenizing verse at index {idx}: {verse}", exc_info=True) # Log traceback
            # Re-raise a more specific error
            raise RuntimeError(f"Tokenization failed for verse at index {idx}: {e}") from e


class BibleInstructionDataset(Dataset):
    """Dataset for instruction fine-tuning with biblical data."""

    def __init__(
        self,
        data_path: str,
        tokenizer: BibleTokenizer, # Type hint should be BibleTokenizer
        max_length: int = 512, # Keep max_length in init, but tokenizer uses its own config
        instruction_types: Optional[List[str]] = None
    ):
        """
        Initialize dataset from instruction data.

        Args:
            data_path: Path to instruction JSON file.
            tokenizer: BibleTokenizer instance to use.
            max_length: Maximum sequence length (Note: BibleTokenizer uses its config's max_tokens).
            instruction_types: If provided, filter by these instruction types.

        Raises:
            FileNotFoundError: If data_path doesn't exist.
            ValueError: If data is invalid or empty.
        """
        self.tokenizer = tokenizer
        # self.max_length = max_length # The tokenizer handles max_length based on its config
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

        filtered_data = data
        if instruction_types:
            filtered_data = [item for item in data if item.get("instruction_type") in instruction_types]

        # Validate required fields and basic content
        valid_data = []
        for item in filtered_data:
            if not all(key in item and isinstance(item[key], str) and item[key].strip() for key in ["instruction", "input", "output"]):
                logger.warning(f"Skipping invalid item missing required fields or empty content: {item}")
                continue
            valid_data.append(item)


        if not valid_data:
            raise ValueError("No valid instruction examples found after filtering")

        logger.info(f"Loaded {len(valid_data)} instruction examples from {data_path}")
        return valid_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized instruction example with proper labels."""
        item = self.data[idx]

        try:
            # Format full text including instruction, input and output
            # Added newline characters for better separation
            text = (
                f"Instruction: {item['instruction'].strip()}\n\n"
                f"Input: {item['input'].strip()}\n\n"
                f"Output: {item['output'].strip()}"
            )

            # Corrected: Call the 'tokenize' method of the BibleTokenizer instance
            # The BibleTokenizer handles max_length, truncation, padding internally
            tokenized = self.tokenizer.tokenize(
                text,
                return_tensors="pt" # Request PyTorch tensors
            )

            if "input_ids" not in tokenized or "attention_mask" not in tokenized:
                 raise ValueError("Tokenizer did not return expected keys (input_ids, attention_mask)")

            # Create labels (-100 for prompt part, actual tokens for output)
            labels = tokenized["input_ids"].clone()

            # Find the token index where the output begins
            # We need to tokenize the prompt part separately to find its length in tokens
            prompt_text = f"Instruction: {item['instruction'].strip()}\n\nInput: {item['input'].strip()}\n\nOutput: "
            prompt_tokenized = self.tokenizer.tokenize(prompt_text, return_tensors="pt")

            # The length of the prompt in tokens is the index where the output tokens start
            output_start_token_index = prompt_tokenized["input_ids"].shape[-1]

            # Set tokens corresponding to the prompt part to -100 (ignore in loss)
            # Ensure the index does not exceed the length of the labels tensor
            if output_start_token_index < labels.shape[-1]:
                labels[0, :output_start_token_index] = -100
            else:
                 # This case means the prompt itself is longer than max_length,
                 # or the output starts exactly at or after max_length.
                 # In this scenario, no output tokens will be included, so all labels should be -100.
                 labels[0, :] = -100
                 logger.warning(f"Prompt length exceeds or equals max_length for item {idx}. No output tokens will be used for loss.")


            # Ensure tensors are squeezed to remove the batch dimension added by return_tensors="pt"
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0),
            }
        except Exception as e:
            # Log the error and the problematic item
            logger.error(f"Error tokenizing instruction at index {idx}: {item}", exc_info=True) # Log traceback
            # Re-raise a more specific error
            raise RuntimeError(f"Tokenization failed for instruction at index {idx}: {e}") from e


def create_bible_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: BibleTokenizer, # Type hint should be BibleTokenizer
    batch_size: int = 8,
    max_length: int = 512, # Keep max_length in args, but dataset uses tokenizer config
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for Bible training and validation with proper collation.

    Args:
        train_path: Path to training Bible data.
        val_path: Path to validation Bible data.
        tokenizer: BibleTokenizer instance to use.
        batch_size: Batch size for DataLoaders.
        max_length: Maximum sequence length (Note: BibleDataset uses tokenizer config).
        num_workers: Number of workers for data loading.

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        RuntimeError: If dataloader creation fails.
    """
    try:
        # Pass max_length to the dataset, although the tokenizer primarily controls it
        train_dataset = BibleDataset(train_path, tokenizer, max_length=max_length)
        val_dataset = BibleDataset(val_path, tokenizer, max_length=max_length)

        # Collate function remains the same as it processes the output of __getitem__
        def collate_fn(batch):
            # Filter out any potential None items if __getitem__ returned None on error (though it raises now)
            # batch = [item for item in batch if item is not None]
            if not batch:
                return {} # Return empty dict for empty batch

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
            pin_memory=True,
            # Add persistent_workers=True if num_workers > 0 and using PyTorch 1.7+ for efficiency
            # persistent_workers=num_workers > 0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle validation data
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            # persistent_workers=num_workers > 0
        )

        logger.info(
            f"Created dataloaders with {len(train_dataset)} train and "
            f"{len(val_dataset)} val examples"
        )
        return train_loader, val_loader

    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}", exc_info=True)
        raise RuntimeError(f"Dataloader creation failed: {e}") from e


def create_instruction_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: BibleTokenizer, # Type hint should be BibleTokenizer
    batch_size: int = 4,
    max_length: int = 512, # Keep max_length in args, but dataset uses tokenizer config
    instruction_types: Optional[List[str]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for instruction fine-tuning with proper collation.

    Args:
        train_path: Path to training instruction data.
        val_path: Path to validation instruction data.
        tokenizer: BibleTokenizer instance to use.
        batch_size: Batch size for DataLoaders.
        max_length: Maximum sequence length (Note: BibleInstructionDataset uses tokenizer config).
        instruction_types: Optional list of instruction types to filter.
        num_workers: Number of workers for data loading.

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        RuntimeError: If dataloader creation fails.
    """
    try:
        # Pass max_length to the dataset, although the tokenizer primarily controls it
        train_dataset = BibleInstructionDataset(
            train_path, tokenizer, max_length=max_length, instruction_types=instruction_types
        )
        val_dataset = BibleInstructionDataset(
            val_path, tokenizer, max_length=max_length, instruction_types=instruction_types
        )

        # Collate function remains the same as it processes the output of __getitem__
        def collate_fn(batch):
            # Filter out any potential None items if __getitem__ returned None on error
            # batch = [item for item in batch if item is not None]
            if not batch:
                return {} # Return empty dict for empty batch

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
            pin_memory=True,
            # persistent_workers=num_workers > 0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle validation data
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            # persistent_workers=num_workers > 0
        )

        logger.info(
            f"Created instruction dataloaders with {len(train_dataset)} train and "
            f"{len(val_dataset)} val examples"
        )
        return train_loader, val_loader

    except Exception as e:
        logger.error(f"Failed to create instruction dataloaders: {e}", exc_info=True)
        raise RuntimeError(f"Instruction dataloader creation failed: {e}") from e


