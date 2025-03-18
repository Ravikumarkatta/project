# src/theology/denominational.py
"""
Denominational variations handling for Bible-AI.

Adjusts text and validation based on denominational preferences.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
from src.utils.logger import get_logger

logger = get_logger("DenominationalAdjuster")


class DenominationalAdjuster:
    """Adjusts text for denominational theological preferences."""

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize denominational adjuster with rules.

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
            self.logger.error(f"Failed to load denominational rules: {e}")
            raise

        self.variations: Dict[str, Dict[str, Any]] = self.rules.get("denominational_variations", {})

    def adjust_for_denomination(self, text: str, denomination: str) -> Dict[str, Any]:
        """
        Adjust text for a specific denomination.

        Args:
            text (str): Text to adjust.
            denomination (str): Denomination (e.g., 'catholic').

        Returns:
            Dict[str, Any]: Adjusted text, validity, and details.

        Raises:
            ValueError: If text is empty.
        """
        text = text.strip().lower()
        if not text:
            self.logger.error("Empty text provided for denominational adjustment")
            raise ValueError("Text cannot be empty")

        adjusted_text = text
        valid = True
        details = []

        for topic, rules in self.variations.items():
            default_term = rules.get("default", "")
            denom_term = rules.get("variations", {}).get(denomination, default_term)

            if default_term.lower() in text and denom_term != default_term:
                adjusted_text = adjusted_text.replace(default_term.lower(), denom_term.lower())
                details.append(f"Adjusted '{default_term}' to '{denom_term}' for {denomination}")
            elif denom_term.lower() not in text and default_term.lower() not in text:
                valid = False
                details.append(f"Expected '{denom_term}' or '{default_term}' for {topic} not found")

        result = {
            "adjusted_text": adjusted_text,
            "valid": valid,
            "details": "; ".join(details) if details else "No adjustments needed"
        }
        self.logger.debug(f"Denominational adjustment for '{denomination}': {result}")
        return result

    def get_supported_denominations(self) -> List[str]:
        """Return list of supported denominations."""
        denominations = set()
        for rules in self.variations.values():
            denominations.update(rules.get("variations", {}).keys())
        return list(denominations)


if __name__ == "__main__":
    adjuster = DenominationalAdjuster()
    sample_text = "We celebrate the Lord's Supper"
    result = adjuster.adjust_for_denomination(sample_text, "catholic")
    print(result)