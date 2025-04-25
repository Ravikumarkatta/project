import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.theology.validator import TheologicalValidator


@pytest.fixture
def sample_rules():
    return {
        "doctrinal": {
            "trinity": {
                "key_statements": [
                    "God exists in three persons: Father, Son, and Holy Spirit",
                    "The Trinity is one God in three divine persons",
                ],
                "keywords": ["Trinity", "Father", "Son", "Holy Spirit"],
            },
            "salvation_by_grace": {
                "key_statements": [
                    "Salvation is by grace through faith alone",
                    "We are saved by God's grace, not by works",
                ],
                "keywords": ["grace", "faith", "saved", "salvation"],
            },
            "general": {
                "key_statements": [
                    "God exists in three persons: Father, Son, and Holy Spirit",
                    "Salvation is by grace through faith alone",
                ],
                "keywords": ["Trinity", "grace", "faith"],
            },
        },
        "heretical": {
            "works_based_salvation": {
                "patterns": [
                    r"earn.*salvation",
                    r"work.*to be saved",
                    r"salvation through (good )?works",
                ]
            },
            "general": {
                "patterns": [
                    r"earn.*salvation",
                    r"work.*to be saved",
                ]
            },
        },
    }


@pytest.fixture
def validator(sample_rules, monkeypatch):
    # Mock the SentenceTransformer to avoid loading actual model
    mock_sentence_transformer = MagicMock()
    mock_sentence_transformer.return_value.encode.return_value = np.array(
        [[0.1, 0.2, 0.3]]
    )

    with patch("sentence_transformers.SentenceTransformer", mock_sentence_transformer):
        # Create validator with mocked model
        validator = TheologicalValidator(model_name="all-MiniLM-L6-v2")

        # Replace the rules with our sample rules
        monkeypatch.setattr(validator, "rules", sample_rules)

        return validator


def test_validate_orthodox_statement(validator):
    text = "God exists in three persons: Father, Son, and Holy Spirit. Salvation is by grace through faith alone."
    score = validator.validate(text)
    assert score >= 0.7  # Should pass minimum doctrinal score


def test_validate_heretical_statement(validator):
    text = "You must earn your salvation through good works."
    score = validator.validate(text)
    assert score <= 0.3  # Should fail due to heretical content


def test_validate_mixed_content(validator):
    text = "While we believe in the Trinity, you must work hard to be saved."
    score = validator.validate(text)
    assert 0.3 <= score <= 0.7  # Should get intermediate score


def test_validate_empty_text(validator):
    assert validator.validate("") == 0.5  # Should return neutral score for empty text


def test_validate_batch(validator):
    texts = [
        "God exists in three persons: Father, Son, and Holy Spirit",
        "You must earn your salvation through works",
        "",
    ]
    scores = validator.validate_batch(texts)
    assert len(scores) == 3
    assert scores[0] >= 0.7  # Orthodox statement
    assert scores[1] <= 0.3  # Heretical statement
    assert scores[2] == 0.5  # Empty text returns neutral score


@patch("sentence_transformers.SentenceTransformer")
def test_theological_embeddings(mock_transformer, validator):
    # Mock cosine_similarity to return a predictable value
    with patch(
        "sklearn.metrics.pairwise.cosine_similarity", return_value=np.array([[0.8]])
    ):
        text1 = "God exists in three persons: Father, Son, and Holy Spirit"
        text2 = "The Trinity is one God in three divine persons"

        # Since we're using mock model, let's test the semantics validation directly
        mock_transformer.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        # Test with a new statement not in rules but semantically similar
        score = validator.validate(
            "The Trinity consists of Father, Son and Holy Spirit"
        )
        assert score > 0.5  # Should recognize as similar to doctrinal statements
