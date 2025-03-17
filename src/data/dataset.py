"""Dataset classes for biblical text data processing."""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import os

logger = logging.getLogger(__name__)

class BiblicalDataset(Dataset):
    """Base dataset class for biblical text data."""
    
    def __init__(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512
    ):
        """
        Initialize dataset with input tensors.
        
        Args:
            input_ids: Input tensor of shape [num_samples, seq_len]
            labels: Target tensor of shape [num_samples, seq_len] 
            attention_mask: Optional attention mask tensor
            max_length: Maximum sequence length
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Validate shapes
        assert input_ids.size() == labels.size(), "Input and label tensors must have same size"
        assert input_ids.size() == attention_mask.size(), "Input and attention mask tensors must have same size"
        
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.input_ids[idx][:self.max_length],
            'labels': self.labels[idx][:self.max_length],
            'attention_mask': self.attention_mask[idx][:self.max_length]
        }


class BibleVerseDataset(BiblicalDataset):
    """Dataset specifically for Bible verses with reference tracking."""
    
    def __init__(
        self,
        verses: Dict[str, Dict[int, Dict[int, str]]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize Bible verse dataset.
        
        Args:
            verses: Nested dict of {book: {chapter: {verse: text}}}
            tokenizer: Tokenizer instance for encoding texts
            max_length: Maximum sequence length
        """
        self.verses = verses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create verse index mapping
        self.verse_indices = []
        for book in verses:
            for chapter in verses[book]:
                for verse in verses[book][chapter]:
                    self.verse_indices.append((book, chapter, verse))
    
    def __len__(self) -> int:
        return len(self.verse_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        book, chapter, verse = self.verse_indices[idx]
        text = self.verses[book][chapter][verse]
        
        # Add reference prefix to text
        reference = f"{book} {chapter}:{verse}"
        full_text = f"{reference} {text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "reference": reference
        }


class CommentaryDataset(BiblicalDataset):
    """Dataset for biblical commentaries with verse alignment."""
    
    def __init__(
        self,
        commentaries: List[Dict[str, Union[str, Dict]]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize commentary dataset.
        
        Args:
            commentaries: List of commentary entries with metadata
            tokenizer: Tokenizer instance for encoding texts
            max_length: Maximum sequence length
        """
        self.commentaries = commentaries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.commentaries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.commentaries[idx]
        
        # Format reference if available
        reference = ""
        if all(k in entry for k in ["book", "chapter", "verse_start"]):
            reference = f"{entry['book']} {entry['chapter']}:{entry['verse_start']}"
            if entry.get("verse_end") and entry["verse_end"] != entry["verse_start"]:
                reference += f"-{entry['verse_end']}"
        
        # Combine reference and content
        full_text = f"{reference} {entry['content']}" if reference else entry['content']
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "source": entry.get("source", "unknown"),
            "reference": reference
        }


class BibleInstructionDataset(Dataset):
    """Dataset for instruction fine-tuning with biblical data."""
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Initialize dataset from instruction data.
        
        Args:
            data_path: Path to instruction JSON file.
            tokenizer: HuggingFace tokenizer to use.
            max_length: Maximum sequence length.
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} instruction examples")
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load instruction data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized instruction example."""
        item = self.data[idx]
        
        # Format as instruction prompt
        instruction = item['instruction']
        input_text = item['input']
        output = item['output']
        
        # Format prompt according to instruction tuning format
        prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: "
        
        # Tokenize prompt
        prompt_tokenized = self.tokenizer(
            prompt, 
            max_length=self.max_length // 2,  # Reserve half length for output
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize output (labels)
        output_tokenized = self.tokenizer(
            output,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Combine input_ids: prompt followed by output
        input_ids = torch.cat([
            prompt_tokenized['input_ids'].squeeze(),
            output_tokenized['input_ids'].squeeze()
        ])[:self.max_length]
        
        # Create attention mask (1 for prompt and output tokens, 0 for padding)
        attention_mask = torch.cat([
            prompt_tokenized['attention_mask'].squeeze(),
            output_tokenized['attention_mask'].squeeze()
        ])[:self.max_length]
        
        # Create labels tensor: -100 for prompt tokens (ignored in loss), actual ids for output
        labels = torch.cat([
            torch.full_like(prompt_tokenized['input_ids'].squeeze(), -100),
            output_tokenized['input_ids'].squeeze()
        ])[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        train_path: Path to training data JSON file.
        val_path: Path to validation data JSON file.
        tokenizer: HuggingFace tokenizer.
        batch_size: Batch size for training.
        max_length: Maximum sequence length.
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = BibleInstructionDataset(train_path, tokenizer, max_length)
    val_dataset = BibleInstructionDataset(val_path, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


def load_datasets(data_path: str) -> Tuple[BiblicalDataset, BiblicalDataset]:
    """
    Load processed datasets and return as BiblicalDataset instances.
    
    Args:
        data_path: Path to the directory containing processed data files.
        
    Returns:
        Tuple (train_dataset, val_dataset) as BiblicalDataset instances.
    """
    data_dir = os.path.abspath(data_path)
    train_file = os.path.join(data_dir, 'train.pt')
    val_file = os.path.join(data_dir, 'val.pt')

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"Processed data files not found in {data_dir}")

    train_data = torch.load(train_file)
    val_data = torch.load(val_file)

    train_dataset = BiblicalDataset(
        input_ids=train_data['input_ids'],
        labels=train_data['labels'],
        attention_mask=train_data['attention_mask']
    )
    val_dataset = BiblicalDataset(
        input_ids=val_data['input_ids'],
        labels=val_data['labels'],
        attention_mask=val_data['attention_mask']
    )

    return train_dataset, val_dataset
