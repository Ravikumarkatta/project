# src/theology/denominational.py
"""
Denominational variations handling for Bible-AI.

Adjusts text and validation based on denominational preferences.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.utils.logger import get_logger

logger = get_logger("DenominationalAdjuster")


class DenominationalAdjuster:
    """Adjusts text for denominational theological preferences."""

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize denominational adjuster with rules.

        Args:
            rules_path (str): Path to theological rules JSON file.
        """
        self.logger = logger
        self.rules = self._load_rules(rules_path)
        self.variations = self.rules.get("denominational_variations", {})
        self.positions = self.rules.get("denominational_positions", {})
        self.sensitivities = self.rules.get("denominational_sensitivities", {})

    def _load_rules(self, rules_path: str) -> Dict[str, Any]:
        """Load theological rules from JSON file."""
        try:
            rules_file = Path(rules_path)
            if not rules_file.exists():
                raise FileNotFoundError(f"Rules file not found: {rules_path}")

            with rules_file.open("r", encoding="utf-8") as f:
                rules = json.load(f)
                self.logger.info(f"Loaded theological rules from {rules_path}")
                return rules

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {rules_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load rules: {e}")
            raise

    def adjust_for_denomination(self, text: str, denomination: str) -> Dict[str, Any]:
        """
        Adjust text for a specific denomination.

        Args:
            text (str): Text to adjust.
            denomination (str): Target denomination.

        Returns:
            Dict[str, Any]: Adjusted text and validation details.
        """
        text = text.lower().strip()
        if not text:
            self.logger.error("Empty text provided for denominational adjustment")
            return {
                "valid": False,
                "details": "No text provided",
                "adjusted_text": "",
                "score": 0.0,
                "issues": ["Empty text"],
            }

        adjusted_text = text
        issues = []
        score = 100.0

        # Check and adjust terminology
        for topic, rules in self.variations.items():
            default_term = rules.get("default", "")
            denom_term = rules.get("variations", {}).get(denomination, default_term)

            if default_term.lower() in text and denom_term != default_term:
                adjusted_text = adjusted_text.replace(
                    default_term.lower(), denom_term.lower()
                )
                issues.append(
                    f"Adjusted '{default_term}' to '{denom_term}' for {denomination}"
                )

        # Check denominational positions
        if denomination in self.positions:
            positions = self.positions[denomination]
            for position, details in positions.items():
                required = details.get("required", [])
                forbidden = details.get("forbidden", [])

                # Check required positions
                for req in required:
                    if not any(phrase.lower() in adjusted_text for phrase in req):
                        issues.append(f"Missing {denomination} position on {position}")
                        score -= 20.0

                # Check forbidden positions
                for forb in forbidden:
                    if any(phrase.lower() in adjusted_text for phrase in forb):
                        issues.append(
                            f"Contains position contrary to {denomination} on {position}"
                        )
                        score -= 30.0

        # Check sensitivity areas
        if denomination in self.sensitivities:
            sensitivities = self.sensitivities[denomination]
            for area, terms in sensitivities.items():
                for term in terms:
                    if term.lower() in adjusted_text:
                        context = self._extract_term_context(adjusted_text, term)
                        issues.append(
                            f"Sensitive term '{term}' used in context: {context}"
                        )
                        score -= 10.0

        # Normalize score
        score = max(0.0, min(100.0, score))

        return {
            "valid": score >= self.rules.get("minimum_score", 70.0),
            "adjusted_text": adjusted_text,
            "score": score,
            "details": "; ".join(issues) if issues else "No adjustments needed",
            "issues": issues,
        }

    def _extract_term_context(self, text: str, term: str, window: int = 50) -> str:
        """Extract context around a term."""
        term_pos = text.find(term.lower())
        if term_pos == -1:
            return ""

        start = max(0, term_pos - window)
        end = min(len(text), term_pos + len(term) + window)
        return f"...{text[start:end]}..."

    def get_denominational_positions(self, denomination: str) -> Dict[str, Any]:
        """Get theological positions for a denomination."""
        return self.positions.get(denomination, {})

    def get_supported_denominations(self) -> List[str]:
        """Get list of supported denominations."""
        denominations = set()
        for rules in self.variations.values():
            denominations.update(rules.get("variations", {}).keys())
        for position in self.positions:
            denominations.add(position)
        return sorted(list(denominations))

    def get_sensitivity_terms(self, denomination: str) -> Dict[str, List[str]]:
        """Get sensitivity terms for a denomination."""
        return self.sensitivities.get(denomination, {})

    def validate_denominational_consistency(
        self, text: str, denomination: str
    ) -> Dict[str, Any]:
        """
        Validate text for denominational consistency.

        Args:
            text (str): Text to validate.
            denomination (str): Denomination to check against.

        Returns:
            Dict[str, Any]: Validation results.
        """
        # First adjust the text
        adjustment_result = self.adjust_for_denomination(text, denomination)

        # Additional consistency checks
        positions = self.get_denominational_positions(denomination)
        sensitivity_terms = self.get_sensitivity_terms(denomination)

        issues = adjustment_result.get("issues", [])
        score = adjustment_result.get("score", 100.0)

        # Check for mixed denominational terminology
        mixed_terms = self._check_mixed_terminology(text, denomination)
        if mixed_terms:
            issues.extend(mixed_terms)
            score -= 15.0 * len(mixed_terms)

        return {
            "valid": score >= self.rules.get("minimum_score", 70.0),
            "score": max(0.0, score),
            "issues": issues,
            "mixed_terminology": mixed_terms,
            "adjusted_text": adjustment_result["adjusted_text"],
        }

    def _check_mixed_terminology(
        self, text: str, primary_denomination: str
    ) -> List[str]:
        """Check for mixed denominational terminology."""
        issues = []
        text = text.lower()

        # Check all variations except the primary denomination
        for topic, rules in self.variations.items():
            variations = rules.get("variations", {})
            primary_term = variations.get(primary_denomination)

            if not primary_term:
                continue

            # Check for terms from other denominations
            for denom, term in variations.items():
                if denom != primary_denomination and term.lower() in text:
                    issues.append(
                        f"Mixed terminology: using '{term}' ({denom}) instead of "
                        f"'{primary_term}' ({primary_denomination})"
                    )

        return issues
