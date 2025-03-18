# src/model/tests/test_verse_detector.py
import pytest
import torch
from src.model.verse_detector import VerseDetector, VerseSegmenter

def test_verse_detector():
    model = VerseDetector(hidden_dim=768, num_verse_types=5)
    input_tensor = torch.randn(2, 10, 768)
    output = model(input_tensor)
    assert output["verse_logits"].shape == (2, 10, 5)
    assert output["verse_features"].shape == (2, 10, 768)

def test_verse_segmenter():
    model = VerseSegmenter(hidden_dim=768)
    input_tensor = torch.randn(2, 10, 768)
    output = model(input_tensor)
    assert output["boundary_scores"].shape == (2, 10, 1)

    tokens = ["John", "3:16", "For", "God", "so", "loved"]
    boundary_scores = torch.tensor([0.1, 0.9, 0.2, 0.3, 0.1, 0.8])
    segments = VerseSegmenter.segment_text(boundary_scores, tokens, threshold=0.5)
    assert segments == ["John 3:16", "For God so loved"]