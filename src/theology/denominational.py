# /workspaces/project/src/theology/denominational.py
"""
Denominational variations handling for Bible-AI.

Adjusts text and validation based on denominational preferences.
"""

import json
import re  # Ensure re is imported
from pathlib import Path
from typing import Any, Dict, List, Set, Union

# Use get_logger if standardizing, otherwise keep setup_logger
from src.utils.logger import setup_logger  # Or get_logger

logger = setup_logger("DenominationalAdjuster")  # Or get_logger


class DenominationalAdjuster:
    """Adjusts text for denominational theological preferences."""

    # Constants for score penalties (Maintainability Improvement)
    _POSITION_REQUIRED_MISSING_PENALTY: float = -20.0
    _POSITION_FORBIDDEN_PRESENT_PENALTY: float = -30.0
    _SENSITIVITY_TERM_PENALTY: float = -10.0
    _MIXED_TERMINOLOGY_PENALTY_PER_TERM: float = -15.0

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize denominational adjuster with rules.

        Args:
            rules_path (str): Path to theological rules JSON file.
        """
        self.logger = logger
        self.rules = self._load_rules(rules_path)
        # Use .get() for safer access to potentially missing top-level keys
        self.variations = self.rules.get("denominational_variations", {})
        self.positions = self.rules.get("denominational_positions", {})
        self.sensitivities = self.rules.get("denominational_sensitivities", {})

    def _load_rules(self, rules_path: str) -> Dict[str, Any]:
        """Load theological rules from JSON file."""
        try:
            rules_file = Path(rules_path)
            # Use is_file() for a more specific check
            if not rules_file.is_file():
                # Log error and return empty if default path, raise if specific path
                if rules_path == "config/theological_rules.json":
                    self.logger.error(
                        f"Default rules file not found or not a file: {rules_path}. Denominational checks may be limited."
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
                    return {}  # Return empty for resilience

        except json.JSONDecodeError:
            self.logger.exception(
                f"Invalid JSON in {rules_path}. Cannot load denominational rules."
            )
            raise  # Re-raise after logging
        except FileNotFoundError as e:
            self.logger.error(str(e))
            raise  # Re-raise FileNotFoundError if specific path was given
        except Exception:
            # Catch other potential errors like permission issues
            self.logger.exception(f"Failed to load or parse rules from {rules_path}")
            raise  # Re-raise other exceptions

    def _check_phrase_list_or_string(
        self, text_lower: str, phrases: Union[str, List[str]]
    ) -> bool:
        """Helper to check if any phrase (or a single string) exists in text using regex."""
        if isinstance(phrases, str) and phrases:
            # Handle single string case
            return bool(re.search(rf"\b{re.escape(phrases.lower())}\b", text_lower))
        elif isinstance(phrases, list):
            # Handle list case
            return any(
                isinstance(p, str)
                and p
                and re.search(rf"\b{re.escape(p.lower())}\b", text_lower)
                for p in phrases
            )
        # If not string or list, or empty list/string, return False
        return False

    def adjust_for_denomination(self, text: str, denomination: str) -> Dict[str, Any]:
        """
        Adjust text for a specific denomination using regex matching.

        Args:
            text (str): Text to adjust.
            denomination (str): Target denomination.

        Returns:
            Dict[str, Any]: Adjusted text and validation details.
        """
        # Use strip() before checking length
        text_processed = text.strip()
        if not text_processed:
            self.logger.warning("Empty text provided for denominational adjustment")
            return {
                "valid": False,
                "details": "No text provided",
                "adjusted_text": "",
                "score": 0.0,
                "issues": ["Empty text"],
            }

        # Work with original case for replacement, but use lower for checks
        adjusted_text = text_processed
        text_lower = text_processed.lower()
        issues = []
        score = 100.0

        # Check and adjust terminology
        # Ensure variations is a dict
        if isinstance(self.variations, dict):
            for topic, rules in self.variations.items():
                # Ensure rules is a dict
                if not isinstance(rules, dict):
                    self.logger.warning(
                        f"Invalid structure for variation rule '{topic}'. Expected dict."
                    )
                    continue

                default_term = rules.get("default", "")
                denom_term = rules.get("variations", {}).get(denomination, default_term)

                # Check if default_term and denom_term are valid strings and different
                if (
                    isinstance(default_term, str)
                    and default_term
                    and isinstance(denom_term, str)
                    and denom_term
                    and denom_term.lower() != default_term.lower()
                ):
                    # Use regex word boundaries for checking presence of default term
                    # Use re.IGNORECASE for replacement to handle original casing better
                    pattern = re.compile(
                        rf"\b{re.escape(default_term)}\b", re.IGNORECASE
                    )
                    if pattern.search(adjusted_text):  # Search in original case text
                        adjusted_text = pattern.sub(
                            denom_term, adjusted_text
                        )  # Replace using denom_term (original case)
                        issues.append(
                            f"Adjusted '{default_term}' to '{denom_term}' for {denomination}"
                        )
                        # Update text_lower if replacement happened
                        text_lower = adjusted_text.lower()
        else:
            self.logger.warning(
                "Denominational variations rules not loaded correctly (not a dict). Skipping terminology adjustment."
            )

        # Check denominational positions
        if not isinstance(self.positions, dict):
            self.logger.warning(
                "Denominational positions rules not loaded correctly (not a dict). Skipping position checks."
            )
        elif denomination in self.positions:
            denom_positions = self.positions[denomination]
            if isinstance(denom_positions, dict):
                for position, details in denom_positions.items():
                    if not isinstance(details, dict):
                        self.logger.warning(
                            f"Invalid structure for position '{position}' for denomination '{denomination}'. Expected dict."
                        )
                        continue
                    required = details.get("required", [])
                    forbidden = details.get("forbidden", [])

                    if required and not self._check_phrase_list_or_string(
                        text_lower, required
                    ):
                        req_name = (
                            required[0]
                            if isinstance(required, list) and required
                            else required
                            if isinstance(required, str)
                            else "requirement"
                        )
                        issues.append(
                            f"Missing {denomination} position on {position} (e.g., '{req_name}')"
                        )
                        score += self._POSITION_REQUIRED_MISSING_PENALTY

                    if forbidden and self._check_phrase_list_or_string(
                        text_lower, forbidden
                    ):
                        forb_name = (
                            forbidden[0]
                            if isinstance(forbidden, list) and forbidden
                            else forbidden
                            if isinstance(forbidden, str)
                            else "forbidden element"
                        )
                        issues.append(
                            f"Contains position contrary to {denomination} on {position} (found related to '{forb_name}')"
                        )
                        score += self._POSITION_FORBIDDEN_PRESENT_PENALTY
            else:
                self.logger.warning(
                    f"Invalid structure for positions for denomination '{denomination}'. Expected dict."
                )
        else:
            self.logger.debug(
                f"No specific positions found for denomination '{denomination}'."
            )

        # Check sensitivity areas
        if not isinstance(self.sensitivities, dict):
            self.logger.warning(
                "Denominational sensitivities rules not loaded correctly (not a dict). Skipping sensitivity checks."
            )
        elif denomination in self.sensitivities:
            denom_sensitivities = self.sensitivities[denomination]
            if isinstance(denom_sensitivities, dict):
                for area, terms in denom_sensitivities.items():
                    if not isinstance(terms, list):
                        self.logger.warning(
                            f"Invalid structure for sensitivity area '{area}' for denomination '{denomination}'. Expected list of terms."
                        )
                        continue

                    for term in terms:
                        if isinstance(term, str) and term:
                            if re.search(rf"\b{re.escape(term.lower())}\b", text_lower):
                                context = self._extract_term_context(
                                    text_lower, term.lower()
                                )
                                issues.append(
                                    f"Sensitive term '{term}' related to '{area}' used in context: {context}"
                                )
                                score += self._SENSITIVITY_TERM_PENALTY
                        else:
                            self.logger.warning(
                                f"Invalid sensitivity term found for area '{area}', denomination '{denomination}': {term}"
                            )
            else:
                self.logger.warning(
                    f"Invalid structure for sensitivities for denomination '{denomination}'. Expected dict."
                )
        else:
            self.logger.debug(
                f"No specific sensitivities found for denomination '{denomination}'."
            )

        # Normalize score
        score = max(0.0, min(100.0, score))

        # Determine overall validity based on score threshold from rules (or default)
        min_score = 70.0  # Default minimum score
        if isinstance(
            self.rules, dict
        ):  # Check if self.rules is a dict before accessing
            min_score = self.rules.get("minimum_score", 70.0)
            # Ensure min_score is a number
            if not isinstance(min_score, (int, float)):
                self.logger.warning(
                    f"Invalid 'minimum_score' in rules config ({min_score}). Using default 70.0."
                )
                min_score = 70.0
        else:
            self.logger.warning(
                "Rules not loaded correctly (not a dict). Using default minimum score 70.0."
            )

        return {
            "valid": score >= min_score,
            "adjusted_text": adjusted_text,  # Return text with original casing preserved where possible
            "score": score,
            "details": "; ".join(issues)
            if issues
            else "No adjustments or issues found",
            "issues": issues,
        }

    def _extract_term_context(
        self, text_lower: str, term_lower: str, window: int = 50
    ) -> str:
        """Extract context around a term using regex (expects lowercased inputs)."""
        try:
            # Use regex to find the term ensuring word boundaries
            match = re.search(rf"\b{re.escape(term_lower)}\b", text_lower)
            if not match:
                self.logger.debug(
                    f"Term '{term_lower}' not found for context extraction."
                )
                return "[term not found]"

            term_pos, term_end_pos = match.span()

            start = max(0, term_pos - window)
            end = min(len(text_lower), term_end_pos + window)

            # Add ellipsis for clarity
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text_lower) else ""

            return f"{prefix}{text_lower[start:end]}{suffix}"
        except Exception as e:
            self.logger.error(
                f"Error extracting context for term '{term_lower}': {e}", exc_info=True
            )
            return "[error extracting context]"

    def get_denominational_positions(self, denomination: str) -> Dict[str, Any]:
        """Get theological positions for a denomination."""
        # Ensure positions is a dict before accessing
        if not isinstance(self.positions, dict):
            self.logger.warning(
                "Denominational positions not loaded correctly, cannot get positions."
            )
            return {}

        result = self.positions.get(denomination)
        if isinstance(result, dict):
            return result
        else:
            self.logger.warning(
                f"Invalid structure for positions for denomination '{denomination}'. Expected dict."
            )
            return {}

    def get_supported_denominations(self) -> List[str]:
        """Get list of supported denominations."""
        denominations: Set[str] = set()
        # Ensure variations and positions are dicts before iterating
        if isinstance(self.variations, dict):
            for rules in self.variations.values():
                if isinstance(rules, dict):  # Check inner rule structure
                    denominations.update(rules.get("variations", {}).keys())
        if isinstance(self.positions, dict):
            denominations.update(self.positions.keys())
        return sorted(list(denominations))

    def get_sensitivity_terms(self, denomination: str) -> Dict[str, List[str]]:
        """Get sensitivity terms for a denomination."""
        # Ensure sensitivities is a dict before accessing
        if isinstance(self.sensitivities, dict):
            # FIX: Removed unnecessary type: ignore
            # Also ensure the returned value for the denom is a dict, default to {}
            denom_sens = self.sensitivities.get(denomination, {})
            return denom_sens if isinstance(denom_sens, dict) else {}
        self.logger.warning(
            "Denominational sensitivities not loaded correctly, cannot get terms."
        )
        return {}

    def validate_denominational_consistency(
        self, text: str, denomination: str
    ) -> Dict[str, Any]:
        """
        Validate text for denominational consistency using regex matching.

        Args:
            text (str): Text to validate.
            denomination (str): Denomination to check against.

        Returns:
            Dict[str, Any]: Validation results.
        """
        # First adjust the text and get initial validation
        adjustment_result = self.adjust_for_denomination(text, denomination)
        adjusted_text = adjustment_result.get("adjusted_text", "")
        issues = adjustment_result.get("issues", [])
        score = adjustment_result.get("score", 100.0)

        # Check for mixed terminology
        mixed_issues = self._check_mixed_terminology(adjusted_text, denomination)
        if mixed_issues:
            issues.extend(mixed_issues)
            # Adjust score for each mixed terminology issue found
            score += len(mixed_issues) * self._MIXED_TERMINOLOGY_PENALTY_PER_TERM

        # Clamp score between 0 and 100
        score = max(0.0, min(100.0, score))

        # Safely determine min_score
        min_score = 70.0
        if isinstance(self.rules, dict):
            maybe_min_score = self.rules.get("minimum_score", 70.0)
            if isinstance(maybe_min_score, (int, float)):
                min_score = maybe_min_score
            else:
                self.logger.warning(
                    f"Invalid 'minimum_score' in rules config ({maybe_min_score}). Using default 70.0."
                )
        else:
            self.logger.warning(
                "Rules not loaded correctly (not a dict). Using default minimum score 70.0."
            )

        # Now return AFTER everything is checked
        return {
            "valid": score >= min_score,
            "adjusted_text": adjusted_text,
            "score": score,
            "details": "; ".join(issues)
            if issues
            else "No adjustments or issues found",
            "issues": issues,
        }

    def _check_mixed_terminology(
        self, text: str, primary_denomination: str
    ) -> List[str]:
        """Check for mixed denominational terminology using regex matching."""
        issues: List[str] = []
        text_lower = text.lower()  # Lowercase once for checking

        # Ensure variations is a dict before iterating
        if not isinstance(self.variations, dict):
            self.logger.warning(
                "Denominational variations rules not loaded correctly (not a dict). Skipping mixed terminology check."
            )
            return issues

        for topic, rules in self.variations.items():
            # Ensure rules is a dict
            if not isinstance(rules, dict):
                self.logger.warning(
                    f"Invalid structure for variation rule '{topic}'. Skipping mixed terminology check for this topic."
                )
                continue

            variations = rules.get("variations", {})
            # Ensure variations is a dict
            if not isinstance(variations, dict):
                self.logger.warning(
                    f"Invalid 'variations' structure within rule '{topic}'. Skipping."
                )
                continue

            primary_term = variations.get(primary_denomination)
            # Ensure primary_term is a valid string
            if not isinstance(primary_term, str) or not primary_term:
                continue  # No term defined for the primary denomination, skip topic

            # Check for terms from other denominations
            for denom, term in variations.items():
                # Ensure term is a valid string and denom is different
                if denom != primary_denomination and isinstance(term, str) and term:
                    # Use regex word boundaries
                    if re.search(rf"\b{re.escape(term.lower())}\b", text_lower):
                        issues.append(
                            f"Mixed terminology: using '{term}' (associated with {denom}) instead of potentially '{primary_term}' (for {primary_denomination}) on topic '{topic}'"
                        )

        return issues
