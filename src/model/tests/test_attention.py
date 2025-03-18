# src/model/tests/test_attention.py
import pytest
import torch
from src.model.attention import MultiHeadAttention, BiblicalSelfAttention

def test_multi_head_attention():
    batch_size, seq_length, hidden_size = 2, 10, 768
    model = MultiHeadAttention(hidden_size=hidden_size, num_attention_heads=12)
    input_tensor = torch.randn(batch_size, seq_length, hidden_size)
    output, _ = model(input_tensor, input_tensor, input_tensor)
    assert output.shape == (batch_size, seq_length, hidden_size)

def test_biblical_self_attention():
    model = BiblicalSelfAttention(hidden_size=768, num_attention_heads=12)
    input_tensor = torch.randn(2, 10, 768)
    verse_positions = torch.zeros(2, 10, 10)
    verse_positions[:, 0, :] = 1  # Simulate verse reference at position 0
    output, _ = model(input_tensor, verse_positions=verse_positions)
    assert output.shape == (2, 10, 768)