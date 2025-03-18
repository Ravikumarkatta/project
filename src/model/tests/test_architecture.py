# src/model/tests/test_architecture.py
import pytest
import torch
from src.model.architecture import BiblicalTransformer, BiblicalTransformerConfig

def test_biblical_transformer():
    config = BiblicalTransformerConfig(vocab_size=1000, hidden_size=768, num_hidden_layers=2)
    model = BiblicalTransformer(config)
    input_ids = torch.randint(0, 1000, (2, 10))
    output = model(input_ids)
    assert output["logits"].shape == (2, 10, 1000)
    assert output["verse_logits"].shape == (2, 10, config.num_bible_books * 200)