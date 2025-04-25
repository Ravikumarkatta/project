import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.data.dataset import (
    BibleDataset,
    BibleInstructionDataset,
    create_bible_dataloaders,
    create_instruction_dataloaders,
)
from src.data.tokenization import BibleTokenizer


@pytest.fixture
def sample_bible_data():
    """Provides sample Bible data for testing."""
    return {
        "Genesis": {
            "1": {
                "1": "In the beginning God created the heaven and the earth.",
                "2": "And the earth was without form, and void.",
            },
            "2": {
                "1": "Thus the heavens and the earth were finished.",
                "2": "And on the seventh day God ended his work.",
            },
        }
    }


@pytest.fixture
def sample_instruction_data():
    """Provides sample instruction data for testing."""
    return [
        {
            "instruction": "Explain the verse",
            "input": "Genesis 1:1",
            "output": "This verse describes the creation of the universe by God.",
            "instruction_type": "explanation",
        },
        {
            "instruction": "Find cross references",
            "input": "Genesis 1:1",
            "output": "John 1:1, Hebrews 11:3",
            "instruction_type": "cross_reference",
        },
    ]


@pytest.fixture
def temp_bible_file(sample_bible_data):
    """Creates a temporary JSON file with sample Bible data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_bible_data, f)
        return Path(f.name)


@pytest.fixture
def temp_instruction_file(sample_instruction_data):
    """Creates a temporary JSON file with sample instruction data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_instruction_data, f)
        return Path(f.name)


@pytest.fixture
def tokenizer():
    """Provides an instance of BibleTokenizer for tests."""
    # Corrected the keyword argument from 'base_tokenizer_name' to 'base_model'
    return BibleTokenizer(base_model="bert-base-uncased")


def test_bible_dataset_initialization(temp_bible_file, tokenizer):
    """Tests the initialization of BibleDataset."""
    dataset = BibleDataset(str(temp_bible_file), tokenizer)
    assert len(dataset) == 4  # Total number of verses in sample data

    # Test data sampling
    sampled_dataset = BibleDataset(str(temp_bible_file), tokenizer, sample_ratio=0.5)
    # Use floor division for sampling to handle odd numbers of verses
    assert len(sampled_dataset) == int(len(dataset) * 0.5)


def test_bible_dataset_getitem(temp_bible_file, tokenizer):
    """Tests retrieving an item from BibleDataset."""
    dataset = BibleDataset(str(temp_bible_file), tokenizer)
    item = dataset[0]

    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert item["input_ids"].dim() == 1
    assert item["attention_mask"].dim() == 1


def test_instruction_dataset_initialization(temp_instruction_file, tokenizer):
    """Tests the initialization of BibleInstructionDataset."""
    dataset = BibleInstructionDataset(str(temp_instruction_file), tokenizer)
    assert len(dataset) == 2  # Total number of instructions in sample data

    # Test instruction type filtering
    filtered_dataset = BibleInstructionDataset(
        str(temp_instruction_file), tokenizer, instruction_types=["explanation"]
    )
    assert len(filtered_dataset) == 1


def test_instruction_dataset_getitem(temp_instruction_file, tokenizer):
    """Tests retrieving an item from BibleInstructionDataset."""
    dataset = BibleInstructionDataset(str(temp_instruction_file), tokenizer)
    item = dataset[0]

    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["input_ids"].dim() == 1
    assert item["attention_mask"].dim() == 1
    assert item["labels"].dim() == 1


def test_create_bible_dataloaders(temp_bible_file, tokenizer):
    """Tests the creation of Bible DataLoaders."""
    # Using same file for train/val in test for simplicity
    train_loader, val_loader = create_bible_dataloaders(
        str(temp_bible_file),
        str(temp_bible_file),
        tokenizer,
        batch_size=2,
    )

    # 4 verses / batch_size 2 = 2 batches
    assert len(train_loader) == 2
    assert len(val_loader) == 2 # Also 2 batches for validation

    # Check the shape of a batch
    batch = next(iter(train_loader))
    assert isinstance(batch, dict)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert batch["input_ids"].dim() == 2  # [batch_size, sequence_length]
    assert batch["attention_mask"].dim() == 2
    assert batch["input_ids"].shape[0] == 2 # Check batch size
    assert batch["attention_mask"].shape[0] == 2 # Check batch size


def test_create_instruction_dataloaders(temp_instruction_file, tokenizer):
    """Tests the creation of instruction DataLoaders."""
    # Using same file for train/val in test for simplicity
    train_loader, val_loader = create_instruction_dataloaders(
        str(temp_instruction_file),
        str(temp_instruction_file),
        tokenizer,
        batch_size=2,
    )

    # 2 instructions / batch_size 2 = 1 batch
    assert len(train_loader) == 1
    assert len(val_loader) == 1 # Also 1 batch for validation

    # Check the shape of a batch
    batch = next(iter(train_loader))
    assert isinstance(batch, dict)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].dim() == 2  # [batch_size, sequence_length]
    assert batch["attention_mask"].dim() == 2
    assert batch["labels"].dim() == 2
    assert batch["input_ids"].shape[0] == 2 # Check batch size
    assert batch["attention_mask"].shape[0] == 2 # Check batch size
    assert batch["labels"].shape[0] == 2 # Check batch size


