# src/theology/pastoral.py
"""
Pastoral sensitivity features for Bible-AI.

Ensures compassionate responses for sensitive topics.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
from src.utils.logger import get_logger

logger = get_logger("PastoralSensitivity")


class PastoralSensitivity:
    """Applies pastoral sensitivity to text outputs."""

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize pastoral sensitivity with rules.

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
            self.logger.error(f"Failed to load pastoral rules: {e}")
            raise

        self.sensitivity_rules: Dict[str, Dict[str, list]] = self.rules.get("pastoral_sensitivity", {}).get("topics", {})

    def apply_sensitivity(self, text: str, topic: str) -> Dict[str, Any]:
        """
        Apply pastoral sensitivity for a topic.

        Args:
            text (str): Text to adjust.
            topic (str): Pastoral topic (e.g., 'grief').

        Returns:
            Dict[str, Any]: Adjusted text and sensitivity status.

        Raises:
            ValueError: If text or topic is empty.
        """
        text = text.strip().lower()
        topic = topic.strip().lower()
        if not text or not topic:
            self.logger.error("Empty text or topic provided for pastoral sensitivity")
            raise ValueError("Text and topic cannot be empty")

        if topic not in self.sensitivity_rules:
            self.logger.warning(f"Unknown pastoral topic: {topic}")
            return {"adjusted_text": text, "sensitive": False, "details": f"Topic '{topic}' not recognized"}

        rules = self.sensitivity_rules[topic]
        encouraged = rules.get("encouraged_phrases", [])
        avoid = rules.get("avoid_phrases", [])

        avoid_found = any(phrase.lower() in text for phrase in avoid)
        encouraged_found = any(phrase.lower() in text for phrase in encouraged)

        if avoid_found:
            result = {
                "adjusted_text": text,
                "sensitive": False,
                "details": f"Avoided phrases detected: {', '.join(p for p in avoid if p.lower() in text)}"
            }
        elif encouraged_found:
            result = {
                "adjusted_text": text,
                "sensitive": True,
                "details": "Encouraged phrases present"
            }
        else:
            result = {
                "adjusted_text": f"{text} Consider {', '.join(encouraged)}.",
                "sensitive": True,
                "details": "Added encouragement for sensitivity"
            }

        self.logger.debug(f"Pastoral sensitivity for '{topic}': {result}")
        return result


if __name__ == "__main__":
    sensitivity = PastoralSensitivity()
    sample_text = "Grief has no hope"
    result = sensitivity.apply_sensitivity(sample_text, "grief")
    print(result)