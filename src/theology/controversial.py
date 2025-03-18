# src/theology/controversial.py
"""
Controversial topics handling for Bible-AI.

Manages sensitive theological topics with neutral responses.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from src.utils.logger import get_logger

logger = get_logger("ControversialHandler")


class ControversialHandler:
    """Handles controversial theological topics with sensitivity."""

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize controversial handler with rules.

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
            self.logger.error(f"Failed to load controversial rules: {e}")
            raise

        # Default topics if not in config
        self.controversial_topics: Dict[str, Dict[str, Any]] = self.rules.get("controversial_topics", {
            "predestination": {
                "keywords": ["elect", "predestined", "free will"],
                "neutral_response": "This topic is interpreted differently across traditions."
            },
            "end_times": {
                "keywords": ["rapture", "tribulation", "millennium"],
                "neutral_response": "Eschatological views vary among scholars."
            }
        })

    def handle_controversy(self, text: str) -> Dict[str, Any]:
        """
        Handle controversial topics in text.

        Args:
            text (str): Text to analyze.

        Returns:
            Dict[str, Any]: Result with adjusted text and topic details.

        Raises:
            ValueError: If text is empty.
        """
        text = text.strip().lower()
        if not text:
            self.logger.error("Empty text provided for controversy handling")
            raise ValueError("Text cannot be empty")

        detected_topics = []
        for topic, details in self.controversial_topics.items():
            if any(keyword.lower() in text for keyword in details["keywords"]):
                detected_topics.append({
                    "topic": topic,
                    "response": details["neutral_response"]
                })

        result = {
            "is_controversial": bool(detected_topics),
            "topics": detected_topics,
            "adjusted_text": text + (" " + " ".join(t["response"] for t in detected_topics) if detected_topics else "")
        }
        self.logger.debug(f"Controversy handling result: {result}")
        return result


if __name__ == "__main__":
    handler = ControversialHandler()
    sample_text = "The rapture will occur soon"
    result = handler.handle_controversy(sample_text)
    print(result)