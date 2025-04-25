# /workspaces/project/src/theology/pastoral.py
"""
Pastoral sensitivity features for Bible-AI.

Ensures responses are pastorally appropriate and sensitive.
"""

import json
import re  # Now used for robust matching
from pathlib import Path

# Removed Set, Tuple as they were unused
from typing import Any, Dict, List, Optional, cast

from src.utils.logger import get_logger

logger = get_logger("PastoralSensitivity")


class PastoralSensitivity:
    """Applies pastoral sensitivity to text outputs."""

    # Constants for score penalties (Maintainability Improvement)
    _ENCOURAGED_COMPLETELY_MISSING_PENALTY: float = -20.0
    _ENCOURAGED_PARTIALLY_MISSING_PENALTY: float = -10.0
    _AVOID_PHRASE_PENALTY: float = -30.0
    _SITUATION_REQUIRED_MISSING_PENALTY: float = -15.0
    _SITUATION_TONE_MISSING_PENALTY: float = -10.0
    _CARE_REQUIRED_MISSING_PENALTY: float = -10.0
    _CARE_AVOID_PRESENT_PENALTY: float = -20.0

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize pastoral sensitivity handler.

        Args:
            rules_path (str): Path to theological rules JSON file.
        """
        self.logger = logger
        self.rules = self._load_rules(rules_path)
        # Use .get() chaining for safer access
        self.sensitivity_topics = self.rules.get("pastoral_sensitivity", {}).get(
            "topics", {}
        )
        self.care_guidelines = self.rules.get("pastoral_care", {})
        self.life_situations = self.rules.get("life_situations", {})

    def _load_rules(self, rules_path: str) -> Dict[str, Any]:
        """Load theological rules from JSON file."""
        try:
            rules_file = Path(rules_path)
            if not rules_file.is_file():  # Check if it's a file specifically
                # Log error instead of raising FileNotFoundError immediately if default path is used
                if rules_path == "config/theological_rules.json":
                    self.logger.error(
                        f"Default rules file not found or not a file: {rules_path}. Pastoral checks may be limited."
                    )
                    return {}  # Return empty dict to allow continuation with no rules
                else:
                    # If a specific path was provided, raise the error
                    raise FileNotFoundError(
                        f"Specified rules file not found or not a file: {rules_path}"
                    )

            with rules_file.open("r", encoding="utf-8") as f:
                rules = json.load(f)
                self.logger.info(f"Loaded theological rules from {rules_path}")
                return cast(Dict[str, Any], rules)

        except json.JSONDecodeError:
            # Use logger.exception to include traceback
            self.logger.exception(
                f"Invalid JSON in {rules_path}. Cannot load pastoral rules."
            )
            raise  # Re-raise after logging
        except FileNotFoundError as e:
            self.logger.error(str(e))
            raise  # Re-raise FileNotFoundError if specific path was given
        except Exception as e:
            # Catch other potential errors like permission issues
            self.logger.exception(
                f"Failed to load or parse rules from {rules_path}: {e}"
            )
            raise  # Re-raise other exceptions

    def analyze_sensitivity(
        self, text: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze text for pastoral sensitivity.

        Args:
            text (str): Text to analyze.
            context (str, optional): Additional context about the situation (e.g., "grief").

        Returns:
            Dict[str, Any]: Analysis results with suggestions.
        """
        if not isinstance(text, str) or not text.strip():
            self.logger.warning(
                "analyze_sensitivity called with empty or invalid text."
            )
            # Return a default structure indicating failure/no analysis
            return {
                "sensitive": False,
                "topics": [],
                "suggestions": ["Input text was empty."],
                "score": 0.0,
            }

        text_lower = text.lower().strip()  # Process once

        detected_topics = []
        suggestions = []
        # Use a high initial score that can only decrease
        score = 100.0

        # Check sensitive topics
        for topic_id, topic_rules in self.sensitivity_topics.items():
            # Ensure topic_rules is a dictionary before proceeding
            if isinstance(topic_rules, dict):
                topic_result = self._analyze_sensitive_topic(
                    text_lower, topic_id, topic_rules
                )
                if topic_result["detected"]:
                    detected_topics.append(topic_result)
                    suggestions.extend(topic_result["suggestions"])
                    score = min(score, topic_result["score"])
            else:
                self.logger.warning(
                    f"Invalid rule structure for topic '{topic_id}'. Expected dict, got {type(topic_rules)}. Skipping."
                )

        # Check life situations
        if context:
            situation_result = self._check_life_situation(text_lower, context)
            # Check if the situation was found and processed
            if situation_result["relevant"]:
                suggestions.extend(situation_result["suggestions"])
                score = min(score, situation_result["score"])

        # Apply care guidelines
        care_result = self._apply_care_guidelines(text_lower)
        # Check if any suggestions were generated (implies rules were applied)
        if care_result["suggestions"]:
            suggestions.extend(care_result["suggestions"])
            score = min(score, care_result["score"])

        return {
            "sensitive": bool(detected_topics),
            "topics": detected_topics,
            "suggestions": list(set(suggestions)),  # Remove duplicates
            "score": score,  # Already clamped between 0 and 100 in helper methods
        }

    def _analyze_sensitive_topic(
        self, text_lower: str, topic_id: str, rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze text for a specific sensitive topic using regex."""
        keywords = rules.get("keywords", [])
        encouraged_phrases = rules.get("encouraged_phrases", [])
        avoid_phrases = rules.get("avoid_phrases", [])

        # Check for topic relevance using regex word boundaries
        detected = False
        matched_keywords = []
        if isinstance(keywords, list):
            for keyword in keywords:
                if isinstance(keyword, str) and keyword:
                    # Robustness: Use regex word boundaries
                    if re.search(rf"\b{re.escape(keyword.lower())}\b", text_lower):
                        detected = True
                        matched_keywords.append(keyword)
        else:
            self.logger.warning(
                f"Keywords for topic '{topic_id}' is not a list. Skipping keyword check."
            )

        if not detected:
            return {
                "topic_id": topic_id,
                "detected": False,
                "score": 100.0,
                "suggestions": [],
            }

        # Analyze handling
        score = 100.0
        suggestions = []

        # Check encouraged phrases using regex word boundaries
        encouraged_count = 0
        if isinstance(encouraged_phrases, list):
            encouraged_count = sum(
                1
                for phrase in encouraged_phrases
                if isinstance(phrase, str)
                and phrase
                and re.search(rf"\b{re.escape(phrase.lower())}\b", text_lower)
            )
            total_encouraged = len(
                [p for p in encouraged_phrases if isinstance(p, str) and p]
            )  # Count valid phrases

            if total_encouraged > 0:
                if encouraged_count == 0:
                    score += self._ENCOURAGED_COMPLETELY_MISSING_PENALTY  # Use constant
                    suggestions.append(
                        f"Consider including encouraging language for {topic_id}"
                    )
                # Penalize less if some, but not enough, are present (e.g., less than half)
                elif encouraged_count < total_encouraged / 2:
                    score += self._ENCOURAGED_PARTIALLY_MISSING_PENALTY  # Use constant
                    suggestions.append(
                        f"Could use more pastoral language for {topic_id}"
                    )
        else:
            self.logger.warning(
                f"Encouraged phrases for topic '{topic_id}' is not a list. Skipping check."
            )

        # Check phrases to avoid using regex word boundaries
        if isinstance(avoid_phrases, list):
            for phrase in avoid_phrases:
                if isinstance(phrase, str) and phrase:
                    # Robustness: Use regex word boundaries
                    if re.search(rf"\b{re.escape(phrase.lower())}\b", text_lower):
                        score += self._AVOID_PHRASE_PENALTY  # Use constant
                        # Extract context using the already lowercased text
                        context_snippet = self._extract_phrase_context(
                            text_lower, phrase.lower()
                        )
                        suggestions.append(
                            f"Avoid using '{phrase}' when discussing {topic_id}. Found in context: {context_snippet}"
                        )
        else:
            self.logger.warning(
                f"Avoid phrases for topic '{topic_id}' is not a list. Skipping check."
            )

        return {
            "topic_id": topic_id,
            "detected": True,
            "matched_keywords": matched_keywords,
            "score": max(0.0, score),  # Clamp score
            "suggestions": suggestions,
        }

    def _check_life_situation(self, text_lower: str, context: str) -> Dict[str, Any]:
        """Check text against specific life situation guidelines using regex."""
        situation = self.life_situations.get(context, {})
        # Check if situation rules exist and are a dictionary
        if not isinstance(situation, dict) or not situation:
            self.logger.debug(
                f"No rules found or invalid structure for life situation context: '{context}'"
            )
            return {"relevant": False, "score": 100.0, "suggestions": []}

        score = 100.0
        suggestions = []

        # Check required elements (assuming 'element' is a list of alternatives)
        required = situation.get("required_elements", [])
        if isinstance(required, list):
            for element_list in required:
                # Check if element_list is a list and contains strings
                if isinstance(element_list, list) and all(
                    isinstance(req, str) for req in element_list
                ):
                    # Robustness: Use regex word boundaries
                    found_alternative = any(
                        re.search(rf"\b{re.escape(req.lower())}\b", text_lower)
                        for req in element_list
                        if req  # Check req is not empty
                    )
                    if not found_alternative:
                        score += (
                            self._SITUATION_REQUIRED_MISSING_PENALTY
                        )  # Use constant
                        # BUG FIX: Check if element_list is non-empty before accessing [0]
                        if element_list:
                            suggestions.append(
                                f"Include language related to '{element_list[0]}' when addressing {context}"
                            )
                        else:
                            suggestions.append(
                                f"Missing required element when addressing {context} (check rule config)"
                            )
                else:
                    self.logger.warning(
                        f"Invalid 'required_elements' item for context '{context}'. Expected list of strings, got: {element_list}"
                    )
        else:
            self.logger.warning(
                f"'required_elements' for context '{context}' is not a list. Skipping check."
            )

        # Check tone requirements (assuming 'tone' is a list of alternatives)
        tone_list = situation.get("tone", [])
        if isinstance(tone_list, list) and tone_list:  # Check if list and not empty
            # Robustness: Use regex word boundaries
            found_tone = any(
                re.search(rf"\b{re.escape(t.lower())}\b", text_lower)
                for t in tone_list
                if isinstance(t, str) and t  # Check t is string and not empty
            )
            if not found_tone:
                score += self._SITUATION_TONE_MISSING_PENALTY  # Use constant
                # BUG FIX: Accessing tone_list[0] is safe now because we checked `if tone_list:`
                suggestions.append(
                    f"Adjust tone to be more '{tone_list[0]}' for {context}"
                )
        elif "tone" in situation and not isinstance(tone_list, list):
            self.logger.warning(
                f"'tone' for context '{context}' is not a list. Skipping check."
            )

        return {"relevant": True, "score": max(0.0, score), "suggestions": suggestions}

    def _apply_care_guidelines(self, text_lower: str) -> Dict[str, Any]:
        """Apply general pastoral care guidelines using regex."""
        score = 100.0
        suggestions = list[str] = []  # Initialize as empty list

        # Ensure care_guidelines is a dictionary
        if not isinstance(self.care_guidelines, dict):
            self.logger.warning(
                f"Care guidelines structure is invalid (not a dict). Skipping application."
            )
            return {"score": score, "suggestions": suggestions}

        for guideline_id, rules in self.care_guidelines.items():
            # Ensure rules for a specific guideline is a dictionary
            if not isinstance(rules, dict):
                self.logger.warning(
                    f"Rule structure for guideline '{guideline_id}' is invalid (not a dict). Skipping."
                )
                continue

            required = rules.get("required", [])
            avoid = rules.get("avoid", [])

            # Check required elements
            if isinstance(required, list) and required:  # Check if list and not empty
                # Robustness: Use regex word boundaries
                found_required = any(
                    re.search(rf"\b{re.escape(req.lower())}\b", text_lower)
                    for req in required
                    if isinstance(req, str) and req
                )
                if not found_required:
                    score += self._CARE_REQUIRED_MISSING_PENALTY  # Use constant
                    suggestions.append(
                        rules.get("suggestion", f"Consider {guideline_id} in response")
                    )
            elif "required" in rules and not isinstance(required, list):
                self.logger.warning(
                    f"'required' for guideline '{guideline_id}' is not a list. Skipping check."
                )

            # Check elements to avoid
            if isinstance(avoid, list) and avoid:  # Check if list and not empty
                # Robustness: Use regex word boundaries
                found_avoid = any(
                    re.search(rf"\b{re.escape(a.lower())}\b", text_lower)
                    for a in avoid
                    if isinstance(a, str) and a
                )
                if found_avoid:
                    score += self._CARE_AVOID_PRESENT_PENALTY  # Use constant
                    suggestions.append(
                        rules.get(
                            "warning", f"Revise approach regarding {guideline_id}"
                        )
                    )
            elif "avoid" in rules and not isinstance(avoid, list):
                self.logger.warning(
                    f"'avoid' for guideline '{guideline_id}' is not a list. Skipping check."
                )

        return {"score": max(0.0, score), "suggestions": suggestions}

    def _extract_phrase_context(
        self, text_lower: str, phrase_lower: str, window: int = 50
    ) -> str:
        """Extract context around a phrase (expects lowercased inputs)."""
        try:
            # Use regex to find the phrase ensuring word boundaries
            match = re.search(rf"\b{re.escape(phrase_lower)}\b", text_lower)
            if not match:
                self.logger.debug(
                    f"Phrase '{phrase_lower}' not found for context extraction."
                )
                return "[phrase not found]"  # Indicate phrase wasn't found

            phrase_pos, phrase_end_pos = match.span()

            start = max(0, phrase_pos - window)
            end = min(len(text_lower), phrase_end_pos + window)

            # Add ellipsis for clarity
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text_lower) else ""

            return f"{prefix}{text_lower[start:end]}{suffix}"
        except Exception as e:
            self.logger.error(
                f"Error extracting context for phrase '{phrase_lower}': {e}",
                exc_info=True,
            )
            return "[error extracting context]"

    # --- Getter Methods ---

    def get_topic_guidelines(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Get pastoral guidelines for a specific topic."""
        # Ensure sensitivity_topics is a dict before accessing
        if isinstance(self.sensitivity_topics, dict):
            return self.sensitivity_topics.get(topic_id)
        self.logger.warning(
            "Sensitivity topics not loaded correctly, cannot get guidelines."
        )
        return None

    def list_sensitive_topics(self) -> List[str]:
        """Get list of sensitive topics."""
        if isinstance(self.sensitivity_topics, dict):
            return list(self.sensitivity_topics.keys())
        self.logger.warning(
            "Sensitivity topics not loaded correctly, cannot list topics."
        )
        return []

    def get_situation_guidelines(self, situation: str) -> Optional[Dict[str, Any]]:
        """Get guidelines for a specific life situation."""
        if isinstance(self.life_situations, dict):
            return self.life_situations.get(situation)
        self.logger.warning(
            "Life situations not loaded correctly, cannot get guidelines."
        )
        return None

    def suggest_pastoral_response(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Get suggested pastoral response elements for a topic."""
        # Ensure sensitivity_topics is a dict before accessing
        if not isinstance(self.sensitivity_topics, dict):
            self.logger.warning(
                "Sensitivity topics not loaded correctly, cannot suggest response."
            )
            return None

        topic = self.sensitivity_topics.get(topic_id)
        # Ensure the retrieved topic is also a dictionary
        if isinstance(topic, dict):
            return {
                "encouraged_phrases": topic.get("encouraged_phrases", []),
                "avoid_phrases": topic.get("avoid_phrases", []),
                "guidance": topic.get("pastoral_guidance", ""),
                "scripture_comfort": topic.get("scripture_comfort", []),
            }
        elif topic_id in self.sensitivity_topics:
            # Log if the key exists but the value isn't a dict
            self.logger.warning(
                f"Rule structure for topic '{topic_id}' is invalid (not a dict). Cannot suggest response."
            )
            return None
        else:
            # Topic ID doesn't exist
            self.logger.debug(f"Topic ID '{topic_id}' not found in sensitivity topics.")
            return None
