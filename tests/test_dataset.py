import pytest
import torch
from pathlib import Path
import json
import tempfile
from src.data.dataset import BibleDataset, BibleInstructionDataset, create_bible_dataloaders, create_instruction_dataloaders 
from src.data.tokenization import BiblicalTokenizer

@pytest.fixture
def sample_bible_data():
    return {
        "Genesis": {
            "1": {
                "1": "In the beginning God created the heaven and the earth.",
                "2": "And the earth was without form, and void."
            },
            "2": {
                "1": "Thus the heavens and the earth were finished.",
                "2": "And on the seventh day God ended his work."
            }
        }
    }

@pytest.fixture
def sample_instruction_data():
    return [
        {
            "instruction": "Explain the verse",
            "input": "Genesis 1:1",
            "output": "This verse describes the creation of the universe by God.",
            "instruction_type": "explanation"
        },
        {
            "instruction": "Find cross references",
            "input": "Genesis 1:1",
            "output": "John 1:1, Hebrews 11:3",
            "instruction_type": "cross_reference"
        }
    ]

@pytest.fixture
def temp_bible_file(sample_bible_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_bible_data, f)
        return Path(f.name)

@pytest.fixture
def temp_instruction_file(sample_instruction_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_instruction_data, f)
        return Path(f.name)

@pytest.fixture
def tokenizer():
    return BiblicalTokenizer(base_tokenizer_name="bert-base-uncased")

def test_bible_dataset_initialization(temp_bible_file, tokenizer):
    dataset = BibleDataset(str(temp_bible_file), tokenizer)
    assert len(dataset) == 4  # Total number of verses in sample data
    
    # Test data sampling
    sampled_dataset = BibleDataset(str(temp_bible_file), tokenizer, sample_ratio=0.5)
    assert len(sampled_dataset) == 2  # Half of the verses

def test_bible_dataset_getitem(temp_bible_file, tokenizer):
    dataset = BibleDataset(str(temp_bible_file), tokenizer)
    item = dataset[0]
    
    assert isinstance(item, dict)
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert isinstance(item['input_ids'], torch.Tensor)
    assert isinstance(item['attention_mask'], torch.Tensor)
    assert item['input_ids'].dim() == 1
    assert item['attention_mask'].dim() == 1

def test_instruction_dataset_initialization(temp_instruction_file, tokenizer):
    dataset = BibleInstructionDataset(str(temp_instruction_file), tokenizer)
    assert len(dataset) == 2  # Total number of instructions in sample data
    
    # Test instruction type filtering
    filtered_dataset = BibleInstructionDataset(
        str(temp_instruction_file),
        tokenizer,
        instruction_types=['explanation']
    )
    assert len(filtered_dataset) == 1

def test_instruction_dataset_getitem(temp_instruction_file, tokenizer):
    dataset = BibleInstructionDataset(str(temp_instruction_file), tokenizer)
    item = dataset[0]
    
    assert isinstance(item, dict)
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'labels' in item
    assert isinstance(item['input_ids'], torch.Tensor)
    assert isinstance(item['attention_mask'], torch.Tensor)
    assert isinstance(item['labels'], torch.Tensor)
    assert item['input_ids'].dim() == 1
    assert item['attention_mask'].dim() == 1
    assert item['labels'].dim() == 1

def test_create_bible_dataloaders(temp_bible_file, tokenizer):
    train_loader, val_loader = create_bible_dataloaders(
        str(temp_bible_file),
        str(temp_bible_file),  # Using same file for train/val in test
        tokenizer,
        batch_size=2
    )
    
    assert len(train_loader) == 2  # 4 verses / batch_size 2
    
    batch = next(iter(train_loader))
    assert isinstance(batch, dict)
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert batch['input_ids'].dim() == 2  # [batch_size, sequence_length]
    assert batch['attention_mask'].dim() == 2

def test_create_instruction_dataloaders(temp_instruction_file, tokenizer):
    train_loader, val_loader = create_instruction_dataloaders(
        str(temp_instruction_file),
        str(temp_instruction_file),  # Using same file for train/val in test
        tokenizer,
        batch_size=2
    )
    
    assert len(train_loader) == 1  # 2 instructions / batch_size 2
    
    batch = next(iter(train_loader))
    assert isinstance(batch, dict)
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels' in batch
    assert batch['input_ids'].dim() == 2  # [batch_size, sequence_length]
    assert batch['attention_mask'].dim() == 2
    assert batch['labels'].dim() == 2