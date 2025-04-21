# src/theology/controversial.py
"""
Controversial topic handling for Bible-AI.

Provides careful handling of sensitive theological topics.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.utils.logger import setup_logger

logger = setup_logger("ControversialHandler")


class ControversialHandler:
    """Handles controversial theological topics with sensitivity."""

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize controversial topic handler.

        Args:
            rules_path (str): Path to theological rules JSON file.
        """
        self.logger = logger
        self.rules = self._load_rules(rules_path)
        self.controversial_topics = self.rules.get("controversial_topics", {})
        self.historical_debates = self.rules.get("historical_debates", {})
        self.modern_issues = self.rules.get("modern_issues", {})

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

    def analyze_topic(
        self, text: str, topic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze text for controversial topics.

        Args:
            text (str): Text to analyze.
            topic_id (str, optional): Specific topic to check.

        Returns:
            Dict[str, Any]: Analysis results.
        """
        text = text.lower().strip()
        if not text:
            return {
                "controversial": False,
                "topics": [],
                "suggestions": [],
                "score": 100.0,
            }

        detected_topics = []
        suggestions = []
        score = 100.0

        # Check specific topic if provided
        if topic_id:
            if topic_id in self.controversial_topics:
                topic_result = self._analyze_single_topic(text, topic_id)
                return { # type: ignore
                    "controversial": topic_result["detected"],
                    "topics": [topic_result] if topic_result["detected"] else [],
                    "suggestions": topic_result["suggestions"],
                    "score": topic_result["score"],
                }
            else:
                self.logger.warning(f"Unknown controversial topic: {topic_id}")
                return { # type: ignore
                    "controversial": False,
                    "topics": [],
                    "suggestions": [f"Unknown topic: {topic_id}"],
                    "score": 100.0,
                }

        # Analyze all topics
        for topic_id, topic_rules in self.controversial_topics.items():
            topic_result = self._analyze_single_topic(text, topic_id)
            if topic_result["detected"]:
                detected_topics.append(topic_result)
                suggestions.extend(topic_result["suggestions"])
                score = min(score, topic_result["score"])

        # Check historical debates
        historical_results = self._check_historical_debates(text)
        if historical_results["detected"]:
            detected_topics.extend(historical_results["topics"])
            suggestions.extend(historical_results["suggestions"])
            score = min(score, historical_results["score"])

        # Check modern issues
        modern_results = self._check_modern_issues(text)
        if modern_results["detected"]:
            detected_topics.extend(modern_results["topics"])
            suggestions.extend(modern_results["suggestions"])
            score = min(score, modern_results["score"])

        return {
            "controversial": bool(detected_topics),
            "topics": detected_topics,
            "suggestions": suggestions,
            "score": score,
        }

    def _analyze_single_topic(self, text: str, topic_id: str) -> Dict[str, Any]:
        """Analyze text for a specific controversial topic."""
        topic_rules = self.controversial_topics[topic_id]
        keywords = topic_rules.get("keywords", [])
        context_rules = topic_rules.get("context", {})
        balanced_view = topic_rules.get("balanced_view", [])

        # Check for topic keywords
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
                "suggestions": [],
            }

        # Analyze handling of the topic
        score = 100.0
        suggestions = []

        # Check for balanced presentation
        balance_score = self._check_balanced_view(text, balanced_view)
        if balance_score < 100:
            score = min(score, balance_score)
            suggestions.append(f"Consider presenting more balanced view of {topic_id}")

        # Check context rules
        context_score = self._check_context_rules(text, context_rules)
        if context_score < 100:
            score = min(score, context_score)
            suggestions.append(f"Improve contextual handling of {topic_id}")

        return {
            "topic_id": topic_id,
            "detected": True,
            "matched_keywords": matched_keywords,
            "score": score,
            "suggestions": suggestions,
        }

    def _check_balanced_view(self, text: str, balanced_view: List[str]) -> float:
        """Check if text presents a balanced view of the topic."""
        if not balanced_view:
            return 100.0

        matched_points = sum(1 for point in balanced_view if point.lower() in text)
        return (matched_points / len(balanced_view)) * 100

    def _check_context_rules(self, text: str, context_rules: Dict[str, Any]) -> float:
        """Check if text follows context rules for controversial topics."""
        if not context_rules:
            return 100.0

        score = 100.0
        required = context_rules.get("required", [])
        forbidden = context_rules.get("forbidden", [])

        # Check required context
        for req in required:
            if not any(term.lower() in text for term in req):
                score -= 20.0

        # Check forbidden context
        for forb in forbidden:
            if any(term.lower() in text for term in forb):
                score -= 30.0

        return max(0.0, score)

    def _check_historical_debates(self, text: str) -> Dict[str, Any]:
        """Check text for historical theological debates."""
        detected_topics = []
        suggestions = []
        score = 100.0

        for debate_id, debate_rules in self.historical_debates.items():
            keywords = debate_rules.get("keywords", [])
            if any(keyword.lower() in text for keyword in keywords):
                historical_context = debate_rules.get("historical_context", [])
                if not all(context.lower() in text for context in historical_context):
                    score = min(score, 70.0)
                    suggestions.append(f"Include historical context for {debate_id}")
                detected_topics.append(
                    {"topic_id": debate_id, "type": "historical_debate"}
                )

        return {
            "detected": bool(detected_topics),
            "topics": detected_topics,
            "suggestions": suggestions,
            "score": score,
        }

    def _check_modern_issues(self, text: str) -> Dict[str, Any]:
        """Check text for modern theological issues."""
        detected_topics = []
        suggestions = []
        score = 100.0

        for issue_id, issue_rules in self.modern_issues.items():
            keywords = issue_rules.get("keywords", [])
            if any(keyword.lower() in text for keyword in keywords):
                pastoral_guidelines = issue_rules.get("pastoral_guidelines", [])
                if not all(
                    guideline.lower() in text for guideline in pastoral_guidelines
                ):
                    score = min(score, 80.0)
                    suggestions.append(f"Consider pastoral implications for {issue_id}")
                detected_topics.append({"topic_id": issue_id, "type": "modern_issue"})

        return {
            "detected": bool(detected_topics),
            "topics": detected_topics,
            "suggestions": suggestions,
            "score": score,
        }

    def get_topic_guidelines(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Get handling guidelines for a specific topic."""
        return self.controversial_topics.get(topic_id) # type: ignore

    def list_controversial_topics(self) -> Dict[str, List[str]]:
        """Get categorized list of controversial topics.""" # type: ignore
        return {
            "traditional": list(self.controversial_topics.keys()),
            "historical": list(self.historical_debates.keys()),
            "modern": list(self.modern_issues.keys()),
        }

    def get_neutral_response(self, topic_id: str) -> Optional[str]:
        """Get neutral response template for a topic."""
        topic = self.controversial_topics.get(topic_id, {})
        return topic.get("neutral_response") # type: ignore
