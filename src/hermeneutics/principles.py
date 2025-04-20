"""
Hermeneutical principles for Bible-AI.

Implements sound biblical interpretation principles.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
import json
import re
from src.utils.logger import get_logger

logger = get_logger("HermeneuticalPrinciples")

class HermeneuticalPrinciples:
    """Applies sound hermeneutical principles to biblical interpretation."""

    def __init__(self, rules_path: str = "config/theological_rules.json") -> None:
        """
        Initialize hermeneutical principles handler.

        Args:
            rules_path (str): Path to theological rules JSON file.
        """
        self.logger = logger
        self.rules = self._load_rules(rules_path)
        self.principles = self.rules.get("hermeneutical_principles", {})
        self.genres = self.rules.get("biblical_genres", {})
        self.contexts = self.rules.get("interpretive_contexts", {})
        
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

    def analyze_interpretation(self, 
                             text: str, 
                             verse_refs: List[str],
                             genre: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze biblical interpretation for hermeneutical soundness.

        Args:
            text (str): The interpretation text to analyze.
            verse_refs (List[str]): Referenced Bible verses.
            genre (str, optional): Biblical genre if known.

        Returns:
            Dict[str, Any]: Analysis results with suggestions.
        """
        text = text.lower().strip()
        if not text or not verse_refs:
            return {
                "valid": False,
                "score": 0.0,
                "issues": ["Missing text or verse references"],
                "suggestions": []
            }

        score = 100.0
        issues = []
        suggestions = []

        # Check basic interpretation principles
        principles_result = self._check_principles(text, verse_refs)
        if principles_result["issues"]:
            issues.extend(principles_result["issues"])
            suggestions.extend(principles_result["suggestions"])
            score = min(score, principles_result["score"])

        # Check genre-specific principles if genre is provided
        if genre:
            genre_result = self._check_genre_principles(text, genre)
            if genre_result["issues"]:
                issues.extend(genre_result["issues"])
                suggestions.extend(genre_result["suggestions"])
                score = min(score, genre_result["score"])

        # Check contextual analysis
        context_result = self._check_contextual_analysis(text, verse_refs)
        if context_result["issues"]:
            issues.extend(context_result["issues"])
            suggestions.extend(context_result["suggestions"])
            score = min(score, context_result["score"])

        return {
            "valid": score >= self.rules.get("minimum_score", 70.0),
            "score": score,
            "issues": issues,
            "suggestions": list(set(suggestions))  # Remove duplicates
        }

    def _check_principles(self, text: str, verse_refs: List[str]) -> Dict[str, Any]:
        """Check adherence to basic hermeneutical principles."""
        issues = []
        suggestions = []
        score = 100.0

        for principle_id, rules in self.principles.items():
            required = rules.get("required_elements", [])
            avoid = rules.get("avoid", [])
            
            # Check required elements
            for req in required:
                if not any(r.lower() in text for r in req):
                    issues.append(f"Missing {principle_id} principle")
                    suggestions.append(rules.get("suggestion", f"Apply {principle_id} principle"))
                    score -= 15.0

            # Check elements to avoid
            for avoid_item in avoid:
                if any(a.lower() in text for a in avoid_item):
                    issues.append(f"Violation of {principle_id} principle")
                    suggestions.append(rules.get("warning", f"Revise {principle_id} application"))
                    score -= 20.0

        # Check for isolated verse interpretation
        if len(verse_refs) == 1 and not self._has_context_references(text):
            issues.append("Possible isolated verse interpretation")
            suggestions.append("Consider broader scriptural context")
            score -= 15.0

        return {
            "score": max(0.0, score),
            "issues": issues,
            "suggestions": suggestions
        }

    def _check_genre_principles(self, text: str, genre: str) -> Dict[str, Any]:
        """Check genre-specific interpretation principles."""
        genre_rules = self.genres.get(genre, {})
        if not genre_rules:
            return {
                "score": 100.0,
                "issues": [],
                "suggestions": []
            }

        issues = []
        suggestions = []
        score = 100.0

        # Check genre-specific requirements
        required = genre_rules.get("required_considerations", [])
        for req in required:
            if not any(r.lower() in text for r in req):
                issues.append(f"Missing {genre} genre consideration")
                suggestions.append(genre_rules.get("suggestion", f"Consider {genre} genre features"))
                score -= 15.0

        # Check genre-specific pitfalls
        pitfalls = genre_rules.get("common_pitfalls", [])
        for pitfall in pitfalls:
            if any(p.lower() in text for p in pitfall):
                issues.append(f"Common {genre} interpretation pitfall")
                suggestions.append(genre_rules.get("warning", f"Avoid common {genre} pitfalls"))
                score -= 20.0

        return {
            "score": max(0.0, score),
            "issues": issues,
            "suggestions": suggestions
        }

    def _check_contextual_analysis(self, text: str, verse_refs: List[str]) -> Dict[str, Any]:
        """Check contextual analysis in interpretation."""
        issues = []
        suggestions = []
        score = 100.0

        for context_type, rules in self.contexts.items():
            required = rules.get("required_elements", [])
            indicators = rules.get("context_indicators", [])

            # Check if context type is addressed
            if not any(ind.lower() in text for ind in indicators):
                issues.append(f"Missing {context_type} context")
                suggestions.append(rules.get("suggestion", f"Consider {context_type} context"))
                score -= 10.0

            # Check required elements for context type
            for req in required:
                if not any(r.lower() in text for r in req):
                    issues.append(f"Incomplete {context_type} analysis")
                    suggestions.append(rules.get("element_suggestion", f"Include {req[0]} in analysis"))
                    score -= 5.0

        return {
            "score": max(0.0, score),
            "issues": issues,
            "suggestions": suggestions
        }

    def _has_context_references(self, text: str) -> bool:
        """Check if text includes references to broader context."""
        context_terms = [
            "context", "surrounding", "chapter", "book",
            "preceding", "following", "passage"
        ]
        return any(term in text.lower() for term in context_terms)

    def get_genre_guidelines(self, genre: str) -> Optional[Dict[str, Any]]:
        """Get interpretation guidelines for a biblical genre."""
        return self.genres.get(genre)

    def list_principles(self) -> List[str]:
        """Get list of hermeneutical principles."""
        return list(self.principles.keys())

    def get_context_requirements(self, context_type: str) -> Optional[Dict[str, Any]]:
        """Get requirements for a specific type of context."""
        return self.contexts.get(context_type)

    def suggest_interpretation_approach(self, 
                                     genre: str, 
                                     context_types: List[str]) -> Dict[str, Any]:
        """
        Get suggested interpretation approach.

        Args:
            genre (str): Biblical genre.
            context_types (List[str]): Types of context to consider.

        Returns:
            Dict[str, Any]: Suggested approach with guidelines.
        """
        genre_rules = self.genres.get(genre, {})
        context_guidelines = {
            ct: self.contexts.get(ct, {})
            for ct in context_types
            if ct in self.contexts
        }

        return {
            "genre_guidelines": genre_rules.get("guidelines", []),
            "context_considerations": {
                ct: rules.get("guidelines", [])
                for ct, rules in context_guidelines.items()
            },
            "common_pitfalls": genre_rules.get("common_pitfalls", []),
            "recommended_steps": genre_rules.get("interpretation_steps", [])
        }