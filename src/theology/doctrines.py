# src/theology/doctrines.py
"""
Core doctrine handling for Bible-AI.

Provides detailed validation for specific theological doctrines.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import re
from src.utils.logger import get_logger

logger = get_logger("DoctrineChecker")


class DoctrineChecker:
    """Validates text against specific theological doctrines."""

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize doctrine checker with rules.

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
            self.logger.error(f"Failed to load doctrine rules: {e}")
            raise

        self.doctrinal_checks: Dict[str, Dict[str, list]] = self.rules.get("doctrinal_checks", {})

    def check_doctrine(self, text: str, doctrine_name: str) -> Dict[str, Any]:
        """
        Validate text for a specific doctrine.

        Args:
            text (str): Text to validate.
            doctrine_name (str): Doctrine to check (e.g., 'trinity').

        Returns:
            Dict[str, Any]: Validation result with 'valid' and 'details'.
        """
        text = text.lower().strip()
        if not text:
            self.logger.error("Empty text provided for doctrine check")
            return {"valid": False, "details": "No text provided"}

        if doctrine_name not in self.doctrinal_checks:
            self.logger.warning(f"Unknown doctrine: {doctrine_name}")
            return {"valid": False, "details": f"Doctrine '{doctrine_name}' not recognized"}

        rules = self.doctrinal_checks[doctrine_name]
        required = rules.get("required_phrases", [])
        forbidden = rules.get("forbidden_phrases", [])

        req_found = any(re.search(rf"\b{phrase.lower()}\b", text) for phrase in required)
        forb_found = any(re.search(rf"\b{phrase.lower()}\b", text) for phrase in forbidden)

        if req_found and not forb_found:
            result = {"valid": True, "details": f"{doctrine_name} affirmed"}
        elif forb_found:
            result = {"valid": False, "details": f"{doctrine_name} violated: forbidden phrases detected"}
        else:
            result = {"valid": False, "details": f"{doctrine_name} not affirmed"}

        self.logger.debug(f"Doctrine check '{doctrine_name}': {result}")
        return result

    def check_all_doctrines(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Check text against all doctrines.

        Args:
            text (str): Text to validate.

        Returns:
            Dict[str, Dict[str, Any]]: Results for each doctrine.
        """
        if not text.strip():
            self.logger.error("Empty text provided for all doctrines check")
            return {}

        results = {name: self.check_doctrine(text, name) for name in self.doctrinal_checks}
        self.logger.debug(f"All doctrines checked: {results}")
        return results


if __name__ == "__main__":
    checker = DoctrineChecker()
    sample_text = "Salvation is through faith in Christ"
    result = checker.check_doctrine(sample_text, "salvation")
    print(result)