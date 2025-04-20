# src/theology/pastoral.py
"""
Pastoral sensitivity features for Bible-AI.

Ensures responses are pastorally appropriate and sensitive.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
import json
import re
from src.utils.logger import get_logger

logger = get_logger("PastoralSensitivity")

class PastoralSensitivity:
    """Applies pastoral sensitivity to text outputs."""

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize pastoral sensitivity handler.

        Args:
            rules_path (str): Path to theological rules JSON file.
        """
        self.logger = logger
        self.rules = self._load_rules(rules_path)
        self.sensitivity_topics = self.rules.get("pastoral_sensitivity", {}).get("topics", {})
        self.care_guidelines = self.rules.get("pastoral_care", {})
        self.life_situations = self.rules.get("life_situations", {})
        
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

    def analyze_sensitivity(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze text for pastoral sensitivity.

        Args:
            text (str): Text to analyze.
            context (str, optional): Additional context about the situation.

        Returns:
            Dict[str, Any]: Analysis results with suggestions.
        """
        text = text.lower().strip()
        if not text:
            return {
                "sensitive": False,
                "topics": [],
                "suggestions": [],
                "score": 100.0
            }

        detected_topics = []
        suggestions = []
        score = 100.0

        # Check sensitive topics
        for topic_id, topic_rules in self.sensitivity_topics.items():
            topic_result = self._analyze_sensitive_topic(text, topic_id, topic_rules)
            if topic_result["detected"]:
                detected_topics.append(topic_result)
                suggestions.extend(topic_result["suggestions"])
                score = min(score, topic_result["score"])

        # Check life situations
        if context:
            situation_result = self._check_life_situation(text, context)
            if situation_result["relevant"]:
                suggestions.extend(situation_result["suggestions"])
                score = min(score, situation_result["score"])

        # Apply care guidelines
        care_result = self._apply_care_guidelines(text)
        if care_result["suggestions"]:
            suggestions.extend(care_result["suggestions"])
            score = min(score, care_result["score"])

        return {
            "sensitive": bool(detected_topics),
            "topics": detected_topics,
            "suggestions": list(set(suggestions)),  # Remove duplicates
            "score": score
        }

    def _analyze_sensitive_topic(self, text: str, topic_id: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text for a specific sensitive topic."""
        keywords = rules.get("keywords", [])
        encouraged_phrases = rules.get("encouraged_phrases", [])
        avoid_phrases = rules.get("avoid_phrases", [])
        
        # Check for topic relevance
        detected = False
        matched_keywords = []
        for keyword in keywords:
            if keyword.lower() in text:
                detected = True
                matched_keywords.append(keyword)

        if not detected:
            return {
                "topic_id": topic_id,
                "detected": False,
                "score": 100.0,
                "suggestions": []
            }

        # Analyze handling
        score = 100.0
        suggestions = []

        # Check encouraged phrases
        encouraged_count = sum(1 for phrase in encouraged_phrases if phrase.lower() in text)
        if encouraged_count == 0:
            score -= 20.0
            suggestions.append(f"Consider including encouraging language for {topic_id}")
        elif encouraged_count < len(encouraged_phrases) / 2:
            score -= 10.0
            suggestions.append(f"Could use more pastoral language for {topic_id}")

        # Check phrases to avoid
        for phrase in avoid_phrases:
            if phrase.lower() in text:
                score -= 30.0
                context = self._extract_phrase_context(text, phrase)
                suggestions.append(f"Avoid using '{phrase}' when discussing {topic_id}")

        return {
            "topic_id": topic_id,
            "detected": True,
            "matched_keywords": matched_keywords,
            "score": max(0.0, score),
            "suggestions": suggestions
        }

    def _check_life_situation(self, text: str, context: str) -> Dict[str, Any]:
        """Check text against specific life situation guidelines."""
        situation = self.life_situations.get(context, {})
        if not situation:
            return {
                "relevant": False,
                "score": 100.0,
                "suggestions": []
            }

        score = 100.0
        suggestions = []

        # Check required elements
        required = situation.get("required_elements", [])
        for element in required:
            if not any(req.lower() in text for req in element):
                score -= 15.0
                suggestions.append(f"Include {element[0]} when addressing {context}")

        # Check tone requirements
        tone = situation.get("tone", [])
        if tone and not any(t.lower() in text for t in tone):
            score -= 10.0
            suggestions.append(f"Adjust tone to be more {tone[0]} for {context}")

        return {
            "relevant": True,
            "score": max(0.0, score),
            "suggestions": suggestions
        }

    def _apply_care_guidelines(self, text: str) -> Dict[str, Any]:
        """Apply general pastoral care guidelines."""
        score = 100.0
        suggestions = []

        for guideline_id, rules in self.care_guidelines.items():
            required = rules.get("required", [])
            avoid = rules.get("avoid", [])

            # Check required elements
            if required and not any(req.lower() in text for req in required):
                score -= 10.0
                suggestions.append(rules.get("suggestion", f"Consider {guideline_id} in response"))

            # Check elements to avoid
            if avoid and any(a.lower() in text for a in avoid):
                score -= 20.0
                suggestions.append(rules.get("warning", f"Revise approach to {guideline_id}"))

        return {
            "score": max(0.0, score),
            "suggestions": suggestions
        }

    def _extract_phrase_context(self, text: str, phrase: str, window: int = 50) -> str:
        """Extract context around a phrase."""
        phrase_pos = text.find(phrase.lower())
        if phrase_pos == -1:
            return ""
            
        start = max(0, phrase_pos - window)
        end = min(len(text), phrase_pos + len(phrase) + window)
        return f"...{text[start:end]}..."

    def get_topic_guidelines(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Get pastoral guidelines for a specific topic."""
        return self.sensitivity_topics.get(topic_id)

    def list_sensitive_topics(self) -> List[str]:
        """Get list of sensitive topics."""
        return list(self.sensitivity_topics.keys())

    def get_situation_guidelines(self, situation: str) -> Optional[Dict[str, Any]]:
        """Get guidelines for a specific life situation."""
        return self.life_situations.get(situation)

    def suggest_pastoral_response(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Get suggested pastoral response for a topic."""
        topic = self.sensitivity_topics.get(topic_id, {})
        return {
            "encouraged_phrases": topic.get("encouraged_phrases", []),
            "avoid_phrases": topic.get("avoid_phrases", []),
            "guidance": topic.get("pastoral_guidance", ""),
            "scripture_comfort": topic.get("scripture_comfort", [])
        } if topic else None