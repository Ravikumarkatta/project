# src/theology/doctrines.py
"""
Core doctrine handling for Bible-AI.

Provides detailed validation for specific theological doctrines.
"""

from typing import Dict, Any, Optional, List, Tuple
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
        """
        self.logger = logger
        self.rules = self._load_rules(rules_path)
        self.doctrinal_checks = self.rules.get("doctrinal_checks", {})
        self.essential_doctrines = set(self.rules.get("essential_doctrines", []))
        
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
            
    def check_doctrine(self, text: str, doctrine_name: str) -> Dict[str, Any]:
        """
        Validate text for a specific doctrine.

        Args:
            text (str): Text to validate.
            doctrine_name (str): Doctrine to check (e.g., 'trinity').

        Returns:
            Dict[str, Any]: Validation result with details.
        """
        text = text.lower().strip()
        if not text:
            self.logger.error("Empty text provided for doctrine check")
            return {
                "valid": False,
                "details": "No text provided",
                "score": 0.0,
                "issues": ["Empty text"]
            }

        if doctrine_name not in self.doctrinal_checks:
            self.logger.warning(f"Unknown doctrine: {doctrine_name}")
            return {
                "valid": False,
                "details": f"Doctrine '{doctrine_name}' not recognized",
                "score": 0.0,
                "issues": [f"Unknown doctrine: {doctrine_name}"]
            }

        rules = self.doctrinal_checks[doctrine_name]
        result = self._validate_doctrine_rules(text, rules)
        
        self.logger.debug(f"Doctrine check '{doctrine_name}': {result}")
        return result

    def _validate_doctrine_rules(self, text: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Apply doctrine validation rules to text."""
        required = rules.get("required_phrases", [])
        forbidden = rules.get("forbidden_phrases", [])
        key_verses = rules.get("key_verses", [])
        context_rules = rules.get("context", {})
        
        issues = []
        score = 100.0
        
        # Check required phrases
        found_required = []
        for phrase in required:
            if re.search(rf"\b{re.escape(phrase.lower())}\b", text):
                found_required.append(phrase)
            else:
                issues.append(f"Missing required phrase: {phrase}")
                score -= 30.0  # Major deduction for missing required phrases

        # Check forbidden phrases
        found_forbidden = []
        for phrase in forbidden:
            if re.search(rf"\b{re.escape(phrase.lower())}\b", text):
                found_forbidden.append(phrase)
                issues.append(f"Contains forbidden phrase: {phrase}")
                score -= 50.0  # Severe deduction for forbidden phrases

        # Check verse references if required
        if rules.get("requires_scripture", False):
            verse_refs = self._extract_verse_references(text)
            if not verse_refs:
                issues.append("Missing scriptural support")
                score -= 20.0

        # Check contextual rules
        for context_type, context_rules in context_rules.items():
            if not self._check_context_rules(text, context_rules):
                issues.append(f"Failed {context_type} context check")
                score -= 10.0

        # Normalize score
        score = max(0.0, min(100.0, score))
        
        return {
            "valid": score >= rules.get("minimum_score", 70.0),
            "score": score,
            "details": "Doctrine check completed",
            "issues": issues,
            "found_required": found_required,
            "found_forbidden": found_forbidden
        }

    def check_all_doctrines(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Check text against all doctrines.

        Args:
            text (str): Text to validate.

        Returns:
            Dict[str, Dict[str, Any]]: Results for each doctrine.
        """
        results = {}
        for name in self.doctrinal_checks:
            results[name] = self.check_doctrine(text, name)
            
        # Calculate overall doctrinal score
        valid_checks = [r for r in results.values() if r["valid"]]
        overall_score = sum(r["score"] for r in results.values()) / len(results) if results else 0.0
        
        results["_summary"] = {
            "valid": len(valid_checks) == len(results),
            "score": overall_score,
            "total_checks": len(results),
            "passed_checks": len(valid_checks)
        }
        
        return results

    def _extract_verse_references(self, text: str) -> List[str]:
        """Extract Bible verse references from text."""
        # Basic regex for verse references (can be enhanced)
        verse_pattern = r'\b(?:\d\s*)?[A-Za-z]+\s*\d+:\d+(?:-\d+)?\b'
        return re.findall(verse_pattern, text)

    def _check_context_rules(self, text: str, rules: Dict[str, Any]) -> bool:
        """Check if text follows contextual rules."""
        required_context = rules.get("required", [])
        forbidden_context = rules.get("forbidden", [])
        
        # Check required context terms
        if required_context:
            found = any(term.lower() in text for term in required_context)
            if not found:
                return False
                
        # Check forbidden context terms
        if forbidden_context:
            found = any(term.lower() in text for term in forbidden_context)
            if found:
                return False
                
        return True

    def get_doctrine_info(self, doctrine_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific doctrine."""
        return self.doctrinal_checks.get(doctrine_name)

    def list_doctrines(self) -> List[str]:
        """List all available doctrines."""
        return list(self.doctrinal_checks.keys())

    def get_essential_doctrines(self) -> List[str]:
        """Get list of essential doctrines."""
        return list(self.essential_doctrines)


if __name__ == "__main__":
    checker = DoctrineChecker()
    sample_text = "Salvation is through faith in Christ"
    result = checker.check_doctrine(sample_text, "salvation")
    print(result)