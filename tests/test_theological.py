import json
from pathlib import Path

import pytest
import torch

from src.theology.validator import TheologicalValidator


@pytest.fixture
def sample_theological_rules():
    return {
        "min_doctrinal_score": 0.7,
        "min_biblical_accuracy": 0.8,
        "semantic_similarity_threshold": 0.85,
        "essential_doctrines": ["trinity", "salvation_by_grace"],
        "heretical_patterns": ["works_based_salvation"],
    }


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
        },
        "heretical": {
            "works_based_salvation": {
                "patterns": [
                    r"earn.*salvation",
                    r"work.*to be saved",
                    r"salvation through (good )?works",
                ]
            }
        },
    }


@pytest.fixture
def rules_dir(tmp_path, sample_rules):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()

    # Create individual rule files
    for category, rules in sample_rules.items():
        for name, content in rules.items():
            rule_file = rules_dir / f"{name}.json"
            with open(rule_file, "w") as f:
                json.dump(content, f)

    return rules_dir


@pytest.fixture
def config_file(tmp_path, sample_theological_rules):
    config_file = tmp_path / "theological_rules.json"
    with open(config_file, "w") as f:
        json.dump(sample_theological_rules, f)
    return config_file


@pytest.fixture
def validator(config_file, monkeypatch):
    # Mock the rules directory path
    def mock_initialize_rules(self):
        return {
            "doctrinal": {
                "trinity": {
                    "key_statements": [
                        "God exists in three persons: Father, Son, and Holy Spirit"
                    ],
                    "keywords": ["Trinity", "Father", "Son", "Holy Spirit"],
                },
                "salvation_by_grace": {
                    "key_statements": ["Salvation is by grace through faith alone"],
                    "keywords": ["grace", "faith", "saved", "salvation"],
                },
            },
            "heretical": {
                "works_based_salvation": {
                    "patterns": [r"earn.*salvation", r"work.*to be saved"]
                }
            },
        }

    monkeypatch.setattr(
        TheologicalValidator, "_initialize_rules", mock_initialize_rules
    )
    return TheologicalValidator(str(config_file))


def test_validate_orthodox_statement(validator):
    text = "God exists in three persons: Father, Son, and Holy Spirit. Salvation is by grace through faith alone."
    score = validator.validate(text)
    assert score >= 0.7  # Should pass minimum doctrinal score


def test_validate_heretical_statement(validator):
    text = "You must earn your salvation through good works."
    score = validator.validate(text)
    assert score < 0.7  # Should fail due to heretical content


def test_validate_mixed_content(validator):
    text = "While we believe in the Trinity, you must work hard to be saved."
    score = validator.validate(text)
    assert 0.3 <= score <= 0.7  # Should get intermediate score


def test_validate_empty_text(validator):
    assert validator.validate("") == 0.0
    assert validator.validate({"text": ""}) == 0.0


def test_validate_batch(validator):
    texts = [
        "God exists in three persons: Father, Son, and Holy Spirit",
        "You must earn your salvation through works",
        "",
    ]
    scores = validator.validate_batch(texts)
    assert len(scores) == 3
    assert scores[0] >= 0.7  # Orthodox statement
    assert scores[1] < 0.7  # Heretical statement
    assert scores[2] == 0.0  # Empty text


def test_theological_embeddings(validator):
    text1 = "God exists in three persons: Father, Son, and Holy Spirit"
    text2 = "The Trinity is one God in three divine persons"

    emb1 = validator._get_text_embedding(text1)
    emb2 = validator._get_text_embedding(text2)

    # Check embedding shapes
    assert emb1.shape == emb2.shape
    assert emb1.dim() == 2

    # Check semantic similarity
    similarity = torch.cosine_similarity(emb1, emb2)
    assert (
        similarity.item() > 0.7
    )  # Similar theological statements should have high similarity
