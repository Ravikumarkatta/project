# src/theology/validator.py
"""
Main theological validation logic for Bible-AI.

Validates model outputs against theological rules, ensuring scriptural integrity.
"""

from typing import Dict, Any, Optional, List
import json
import re
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("TheologicalValidator")


class TheologicalValidator:
    """Validates text against theological rules with robust scoring."""

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize validator with theological rules.

        Args:
            rules_path (str): Path to theological rules JSON file.

        Raises:
            FileNotFoundError: If rules file is missing.
            json.JSONDecodeError: If rules file is invalid JSON.
        """
        self.logger = logger
        try:
            rules_file = Path(rules_path)
            if not rules_file.exists():
                raise FileNotFoundError(f"Theological rules file not found: {rules_path}")
            with rules_file.open("r", encoding="utf-8") as f:
                self.rules: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {rules_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load theological rules: {e}")
            raise

        self.min_score: float = self.rules.get("minimum_score", 0.9)
        self.core_terms: Dict[str, List[str]] = self.rules.get("core_terms", {})
        self.doctrinal_checks: Dict[str, Dict[str, List[str]]] = self.rules.get("doctrinal_checks", {})
        self.context_sensitive: Dict[str, Dict[str, List[str]]] = self.core_terms.get("context_sensitive", {})

    def validate(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate model output against theological rules.

        Args:
            output (Dict[str, Any]): Model output with 'text' key.

        Returns:
            Dict[str, float]: Validation scores (overall and component-wise).

        Raises:
            ValueError: If output lacks 'text' key or is empty.
        """
        text = output.get("text", "").strip().lower()
        if not text:
            self.logger.error("No valid text provided for validation")
            raise ValueError("Output must contain non-empty 'text' key")

        scores = {
            "overall": 0.0,
            "core_terms": 0.0,
            "doctrines": 0.0,
            "context_sensitivity": 0.0
        }

        # Component scores
        scores["core_terms"] = self._check_core_terms(text)
        scores["doctrines"] = self._check_doctrines(text)
        scores["context_sensitivity"] = self._check_context_sensitivity(text)

        # Overall score as weighted average (equal weights for simplicity)
        scores["overall"] = sum(scores.values()) / len(scores)
        self.logger.debug(f"Validation scores for '{text[:50]}...': {scores}")
        return scores

    def _check_core_terms(self, text: str) -> float:
        """Check presence of required and absence of forbidden terms."""
        required = self.core_terms.get("required", [])
        forbidden = self.core_terms.get("forbidden", [])

        required_score = sum(1 for term in required if term.lower() in text) / max(len(required), 1)
        forbidden_penalty = sum(1 for term in forbidden if term.lower() in text) * 0.2

        score = max(0.0, required_score - forbidden_penalty)
        self.logger.debug(f"Core terms score: {score}")
        return score

    def _check_doctrines(self, text: str) -> float:
        """Verify adherence to core doctrines."""
        score = 0.0
        num_checks = max(len(self.doctrinal_checks), 1)

        for doctrine, rules in self.doctrinal_checks.items():
            required = rules.get("required_phrases", [])
            forbidden = rules.get("forbidden_phrases", [])

            req_found = any(re.search(rf"\b{phrase.lower()}\b", text) for phrase in required)
            forb_found = any(re.search(rf"\b{phrase.lower()}\b", text) for phrase in forbidden)

            if req_found and not forb_found:
                score += 1.0
            elif forb_found:
                score -= 0.5

        final_score = max(0.0, score / num_checks)
        self.logger.debug(f"Doctrines score: {final_score}")
        return final_score

    def _check_context_sensitivity(self, text: str) -> float:
        """Check context-sensitive terms for appropriate usage."""
        score = 0.0
        num_terms = max(len(self.context_sensitive), 1)

        for term, contexts in self.context_sensitive.items():
            if term.lower() in text:
                pos_contexts = contexts.get("positive_contexts", [])
                neg_contexts = contexts.get("negative_contexts", [])

                pos_found = any(ctx.lower() in text for ctx in pos_contexts)
                neg_found = any(ctx.lower() in text for ctx in neg_contexts)

                if pos_found and not neg_found:
                    score += 1.0
                elif neg_found:
                    score -= 0.5

        final_score = max(0.0, score / num_terms)
        self.logger.debug(f"Context sensitivity score: {final_score}")
        return final_score


if __name__ == "__main__":
    validator = TheologicalValidator()
    sample_output = {"text": "God loves us through Jesus and grace"}
    scores = validator.validate(sample_output)
    print(f"Validation scores: {scores}")