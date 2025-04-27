# /workspaces/project/src/theology/doctrines.py
"""
Core doctrine handling for Bible-AI.

Provides detailed validation for specific theological doctrines.
"""

import json
import re  # Ensure re is imported
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger  # Or get_logger if standardizing

logger = setup_logger("DoctrineChecker")  # Or get_logger


class DoctrineChecker:
    """Validates text against specific theological doctrines."""

    # Constants for score penalties (Maintainability Improvement)
    _REQUIRED_PHRASE_MISSING_PENALTY: float = -30.0
    _FORBIDDEN_PHRASE_PRESENT_PENALTY: float = -50.0
    _SCRIPTURE_MISSING_PENALTY: float = -20.0
    _CONTEXT_FAILED_PENALTY: float = -10.0

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize doctrine checker with rules.

        Args:
            rules_path (str): Path to theological rules JSON file.
        """
        self.logger = logger
        self.rules = self._load_rules(rules_path)
        self.doctrinal_checks = self.rules.get("doctrinal_checks", {})
        # Ensure essential_doctrines is always a set, even if missing/null in JSON
        essential_doctrines_list = self.rules.get("essential_doctrines", [])
        self.essential_doctrines = (
            set(essential_doctrines_list)
            if isinstance(essential_doctrines_list, list)
            else set()
        )
        if not isinstance(essential_doctrines_list, list):
            self.logger.warning(
                "'essential_doctrines' key in rules file is not a list. No essential doctrines loaded."
            )

    def _load_rules(self, rules_path: str) -> Dict[str, Any]:
        """Load theological rules from JSON file."""
        try:
            rules_file = Path(rules_path)
            # Use is_file() for a more specific check
            if not rules_file.is_file():
                # Log error and return empty if default path, raise if specific path
                if rules_path == "config/theological_rules.json":
                    self.logger.error(
                        f"Default rules file not found or not a file: {rules_path}. Doctrine checks may be limited."
                    )
                    return {}
                else:
                    raise FileNotFoundError(
                        f"Specified rules file not found or not a file: {rules_path}"
                    )

            with rules_file.open("r", encoding="utf-8") as f:
                rules_data = json.load(f)
                # Add type check for safety before returning
                if isinstance(rules_data, dict):
                    self.logger.info(f"Loaded theological rules from {rules_path}")
                    return rules_data
                else:
                    self.logger.error(
                        f"Invalid JSON structure in {rules_path}: Expected a dictionary (object) at the root."
                    )
                    # Decide behavior: raise error or return empty? Let's return empty for resilience.
                    return {}

        except json.JSONDecodeError:
            self.logger.exception(
                f"Invalid JSON in {rules_path}. Cannot load doctrine rules."
            )
            raise  # Re-raise after logging
        except FileNotFoundError as e:
            self.logger.error(str(e))
            raise  # Re-raise FileNotFoundError if specific path was given
        except Exception:
            # Catch other potential errors like permission issues
            self.logger.exception(f"Failed to load or parse rules from {rules_path}")
            raise  # Re-raise other exceptions

    def check_doctrine(self, text: str, doctrine_name: str) -> Dict[str, Any]:
        """
        Validate text for a specific doctrine.

        Args:
            text (str): Text to validate.
            doctrine_name (str): Doctrine to check (e.g., 'trinity').

        Returns:
            Dict[str, Any]: Validation result with details.
        """
        # Use strip() before checking length
        text_processed = text.lower().strip()
        if not text_processed:
            self.logger.warning("Empty text provided for doctrine check")
            return {
                "valid": False,
                "details": "No text provided",
                "score": 0.0,
                "issues": ["Empty text"],
            }

        # Check if doctrinal_checks is usable
        if not isinstance(self.doctrinal_checks, dict):
            self.logger.error(
                "Doctrinal checks rules were not loaded correctly (not a dict). Cannot perform check."
            )
            return {
                "valid": False,
                "details": "Internal error: Doctrinal rules not loaded.",
                "score": 0.0,
                "issues": ["Internal configuration error"],
            }

        if doctrine_name not in self.doctrinal_checks:
            self.logger.warning(f"Unknown doctrine requested: {doctrine_name}")
            return {
                "valid": False,
                "details": f"Doctrine '{doctrine_name}' not recognized",
                "score": 0.0,
                "issues": [f"Unknown doctrine: {doctrine_name}"],
            }

        rules = self.doctrinal_checks[doctrine_name]
        # Ensure rules for the specific doctrine is a dictionary
        if not isinstance(rules, dict):
            self.logger.error(
                f"Rule structure for doctrine '{doctrine_name}' is invalid (not a dict). Cannot perform check."
            )
            return {
                "valid": False,
                "details": f"Internal error: Invalid rule structure for doctrine '{doctrine_name}'.",
                "score": 0.0,
                "issues": [
                    f"Internal configuration error for doctrine '{doctrine_name}'"
                ],
            }

        result = self._validate_doctrine_rules(text_processed, rules)

        self.logger.debug(
            f"Doctrine check '{doctrine_name}': Score={result['score']}, Valid={result['valid']}"
        )
        return result

    def _validate_doctrine_rules(
        self, text_lower: str, rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply doctrine validation rules to text."""
        # Use .get with default empty list for safety
        required = rules.get("required_phrases", [])
        forbidden = rules.get("forbidden_phrases", [])
        key_verses = rules.get(
            "key_verses", []
        )  # Not currently used in scoring, but available
        context_rules = rules.get("context", {})

        issues = []
        score = 100.0

        # Check required phrases
        found_required = []
        if isinstance(required, list):
            for phrase in required:
                # Ensure phrase is a non-empty string
                if isinstance(phrase, str) and phrase:
                    if re.search(rf"\b{re.escape(phrase.lower())}\b", text_lower):
                        found_required.append(phrase)
                    else:
                        issues.append(f"Missing required phrase: {phrase}")
                        score += self._REQUIRED_PHRASE_MISSING_PENALTY  # Use constant
                else:
                    self.logger.warning(
                        f"Invalid item in 'required_phrases': {phrase}. Skipping."
                    )
        else:
            self.logger.warning(
                "'required_phrases' is not a list in rules. Skipping check."
            )

        # Check forbidden phrases
        found_forbidden = []
        if isinstance(forbidden, list):
            for phrase in forbidden:
                # Ensure phrase is a non-empty string
                if isinstance(phrase, str) and phrase:
                    if re.search(rf"\b{re.escape(phrase.lower())}\b", text_lower):
                        found_forbidden.append(phrase)
                        issues.append(f"Contains forbidden phrase: {phrase}")
                        score += self._FORBIDDEN_PHRASE_PRESENT_PENALTY  # Use constant
                else:
                    self.logger.warning(
                        f"Invalid item in 'forbidden_phrases': {phrase}. Skipping."
                    )
        else:
            self.logger.warning(
                "'forbidden_phrases' is not a list in rules. Skipping check."
            )

        # Check verse references if required
        if rules.get("requires_scripture", False):
            verse_refs = self._extract_verse_references(text_lower)
            if not verse_refs:
                issues.append("Missing scriptural support")
                score += self._SCRIPTURE_MISSING_PENALTY  # Use constant

        # Check contextual rules
        # Ensure context_rules is a dict
        if isinstance(context_rules, dict):
            for context_type, specific_context_rules in context_rules.items():
                # Ensure specific_context_rules is also a dict
                if isinstance(specific_context_rules, dict):
                    if not self._check_context_rules(
                        text_lower, specific_context_rules
                    ):
                        issues.append(f"Failed {context_type} context check")
                        score += self._CONTEXT_FAILED_PENALTY  # Use constant
                else:
                    self.logger.warning(
                        f"Invalid structure for context rule '{context_type}'. Expected dict, got {type(specific_context_rules)}."
                    )
        elif "context" in rules:  # Log warning only if key exists but isn't a dict
            self.logger.warning(
                "'context' rules structure is invalid (not a dict). Skipping context checks."
            )

        # Normalize score
        score = max(0.0, min(100.0, score))

        return {
            "valid": score
            >= rules.get("minimum_score", 70.0),  # Use default minimum score
            "score": score,
            "details": "Doctrine check completed",  # Simple detail message
            "issues": issues,
            "found_required": found_required,
            "found_forbidden": found_forbidden,
        }

    def check_all_doctrines(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Check text against all doctrines.

        Args:
            text (str): Text to validate.

        Returns:
            Dict[str, Dict[str, Any]]: Results for each doctrine, plus a '_summary'.
        """
        results = {}
        # Ensure doctrinal_checks is a dict before iterating
        if not isinstance(self.doctrinal_checks, dict):
            self.logger.error(
                "Doctrinal checks rules were not loaded correctly (not a dict). Cannot perform check_all_doctrines."
            )
            results["_summary"] = {
                "valid": False,
                "score": 0.0,
                "total_checks": 0,
                "passed_checks": 0,
                "error": "Internal configuration error: Doctrinal rules not loaded.",
            }
            return results

        for name in self.doctrinal_checks:
            results[name] = self.check_doctrine(text, name)

        # Calculate overall doctrinal score and summary
        num_checks = len(results)
        if num_checks > 0:
            valid_checks = [
                r for r in results.values() if bool(r.get("valid", False))
            ]  # Safer access
            overall_score = (
                sum(r.get("score", 0.0) for r in results.values()) / num_checks
            )  # Safer access
            passed_count = len(valid_checks)
            all_passed = passed_count == num_checks
        else:
            # Handle case where there are no doctrines to check
            overall_score = 100.0  # Or 0.0 depending on desired behavior? Let's say 100 if nothing to check.
            passed_count = 0
            all_passed = True  # Vacuously true

        results["_summary"] = {
            "valid": all_passed,
            "score": overall_score,
            "total_checks": num_checks,
            "passed_checks": passed_count,
        }

        return results

    def _extract_verse_references(self, text_lower: str) -> List[str]:
        """Extract Bible verse references from text (expects lowercased text)."""
        # Basic regex for verse references (can be enhanced)
        # Handles optional leading digit, book name (simple), chapter:verse, optional range
        # Note: Book names with spaces might still be tricky.
        verse_pattern = r"\b(?:\d\s*)?[a-z]+\s*\d+:\d+(?:-\d+)?\b"
        # Use findall on the already lowercased text
        return re.findall(verse_pattern, text_lower)

    def _check_context_rules(self, text_lower: str, rules: Dict[str, Any]) -> bool:
        """Check if text follows contextual rules using regex (expects lowercased text)."""
        # Use .get with default empty list for safety
        required_context = rules.get("required", [])
        forbidden_context = rules.get("forbidden", [])

        # Check required context terms using regex word boundaries
        if isinstance(required_context, list) and required_context:
            found_required = False
            for term in required_context:
                if isinstance(term, str) and term:
                    # FIX: Use regex word boundaries
                    if re.search(rf"\b{re.escape(term.lower())}\b", text_lower):
                        found_required = True
                        break  # Found one, no need to check others in this list
            if not found_required:
                return False  # Did not find any of the required context terms
        elif "required" in rules and not isinstance(required_context, list):
            self.logger.warning(
                "Context 'required' rule is not a list. Skipping check."
            )

        # Check forbidden context terms using regex word boundaries
        if isinstance(forbidden_context, list) and forbidden_context:
            for term in forbidden_context:
                if isinstance(term, str) and term:
                    # FIX: Use regex word boundaries
                    if re.search(rf"\b{re.escape(term.lower())}\b", text_lower):
                        return False  # Found a forbidden context term
        elif "forbidden" in rules and not isinstance(forbidden_context, list):
            self.logger.warning(
                "Context 'forbidden' rule is not a list. Skipping check."
            )

        return True  # Passed all context checks

    def get_doctrine_info(self, doctrine_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific doctrine."""
        # Ensure doctrinal_checks is a dict before accessing
        if isinstance(self.doctrinal_checks, dict):
            # FIX: Removed unnecessary type: ignore
            return self.doctrinal_checks.get(doctrine_name)
        self.logger.warning(
            "Doctrinal checks not loaded correctly, cannot get doctrine info."
        )
        return None

    def list_doctrines(self) -> List[str]:
        """List all available doctrines."""
        if isinstance(self.doctrinal_checks, dict):
            return list(self.doctrinal_checks.keys())
        self.logger.warning(
            "Doctrinal checks not loaded correctly, cannot list doctrines."
        )
        return []

    def get_essential_doctrines(self) -> List[str]:
        """Get list of essential doctrines."""
        # self.essential_doctrines is guaranteed to be a set by __init__
        return sorted(list(self.essential_doctrines))


# Keep the example usage block
if __name__ == "__main__":
    # Example: Create dummy rules if they don't exist for testing
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    rules_file_path = config_dir / "theological_rules.json"

    if not rules_file_path.exists():
        print(f"Creating dummy rules file: {rules_file_path}")
        dummy_rules_content = {
            "essential_doctrines": ["salvation", "trinity"],
            "doctrinal_checks": {
                "salvation": {
                    "required_phrases": ["faith", "grace", "Christ"],
                    "forbidden_phrases": ["works righteousness", "earn salvation"],
                    "requires_scripture": True,
                    "minimum_score": 60.0,
                },
                "trinity": {
                    "required_phrases": ["Father", "Son", "Holy Spirit", "one God"],
                    "forbidden_phrases": ["modes", "manifestations"],
                    "requires_scripture": False,
                    "minimum_score": 70.0,
                },
            }
            # Add other sections if needed by other modules
        }
        with rules_file_path.open("w", encoding="utf-8") as f:
            json.dump(dummy_rules_content, f, indent=2)

    # Now run the checker
    try:
        checker = DoctrineChecker()  # Will load from default path

        print("\n--- Checking Good Salvation Text ---")
        sample_text_good = "Salvation is through faith alone in Jesus Christ, by God's grace. See Ephesians 2:8."
        result_good = checker.check_doctrine(sample_text_good, "salvation")
        print(json.dumps(result_good, indent=2))

        print("\n--- Checking Bad Salvation Text ---")
        sample_text_bad = "You must earn salvation through good works and faith."
        result_bad = checker.check_doctrine(sample_text_bad, "salvation")
        print(json.dumps(result_bad, indent=2))

        print("\n--- Checking Missing Scripture Text ---")
        sample_text_no_scripture = "Salvation is by faith and grace in Christ."
        result_no_scripture = checker.check_doctrine(
            sample_text_no_scripture, "salvation"
        )
        print(json.dumps(result_no_scripture, indent=2))

        print("\n--- Checking Trinity Text ---")
        sample_text_trinity = "We believe in one God: Father, Son, and Holy Spirit."
        result_trinity = checker.check_doctrine(sample_text_trinity, "trinity")
        print(json.dumps(result_trinity, indent=2))

        print("\n--- Checking All Doctrines ---")
        all_results = checker.check_all_doctrines(sample_text_good)
        print(json.dumps(all_results, indent=2))

        print("\n--- Listing Doctrines ---")
        print("Available:", checker.list_doctrines())
        print("Essential:", checker.get_essential_doctrines())

    except FileNotFoundError:
        print(
            f"ERROR: Could not find or create the rules file at {rules_file_path}. Cannot run example."
        )
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
