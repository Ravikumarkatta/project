"""
Theological verification utilities for Bible-AI.

This module provides functions and utilities to verify theological accuracy
of AI responses and ensure they align with sound biblical doctrine across
different theological traditions.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Union, Set, Any
from pathlib import Path

# Try to import the logger we just created
try:
    from src.utils.logger import get_logger
except ImportError:
    # Fallback if the import path is different
    try:
        from utils.logger import get_logger
    except ImportError:
        import logging
        # Simple logger fallback if our custom logger isn't available
        get_logger = lambda name: logging.getLogger(name)

# Initialize module logger
logger = get_logger("theological_checks")


class TheologicalChecker:
    """
    Provides methods to check theological accuracy of text content.
    This class helps ensure that AI responses align with sound biblical
    doctrine while respecting denominational distinctives.
    """
    
    def __init__(self, 
                 rules_path: Optional[str] = None,
                 doctrines_path: Optional[str] = None,
                 denominations_path: Optional[str] = None):
        """
        Initialize the theological checker with rule sets.
        
        Args:
            rules_path: Path to theological rules JSON file
            doctrines_path: Path to doctrines definitions JSON file
            denominations_path: Path to denominational positions JSON file
        """
        self.rules = {}
        self.doctrines = {}
        self.denominational_positions = {}
        self.essential_doctrines = set()
        self.load_theological_data(rules_path, doctrines_path, denominations_path)
        
        # Dictionary of regex patterns for quick lookup
        self.pattern_cache = {}
        
    def load_theological_data(self, 
                             rules_path: Optional[str] = None,
                             doctrines_path: Optional[str] = None,
                             denominations_path: Optional[str] = None):
        """
        Load theological rules and reference data from JSON files.
        
        Args:
            rules_path: Path to theological rules JSON file
            doctrines_path: Path to doctrines definitions JSON file
            denominations_path: Path to denominational positions JSON file
        """
        # Default paths if not provided
        base_path = Path(os.path.abspath(__file__)).parent.parent.parent
        config_path = base_path / "config"
        
        rules_path = rules_path or str(config_path / "theological_rules.json")
        
        # Load theological rules
        try:
            if os.path.exists(rules_path):
                with open(rules_path, 'r', encoding='utf-8') as f:
                    self.rules = json.load(f)
                    logger.info(f"Loaded {len(self.rules)} theological rules")
                    
                    # Extract essential doctrines
                    if "essential_doctrines" in self.rules:
                        self.essential_doctrines = set(self.rules.get("essential_doctrines", []))
                        logger.info(f"Identified {len(self.essential_doctrines)} essential doctrines")
        except Exception as e:
            logger.error(f"Failed to load theological rules from {rules_path}: {str(e)}")
            
        # Load doctrines if path provided
        if doctrines_path and os.path.exists(doctrines_path):
            try:
                with open(doctrines_path, 'r', encoding='utf-8') as f:
                    self.doctrines = json.load(f)
                    logger.info(f"Loaded {len(self.doctrines)} doctrine definitions")
            except Exception as e:
                logger.error(f"Failed to load doctrines from {doctrines_path}: {str(e)}")
        
        # Load denominational positions if path provided
        if denominations_path and os.path.exists(denominations_path):
            try:
                with open(denominations_path, 'r', encoding='utf-8') as f:
                    self.denominational_positions = json.load(f)
                    logger.info(f"Loaded positions for {len(self.denominational_positions)} denominations")
            except Exception as e:
                logger.error(f"Failed to load denominational positions from {denominations_path}: {str(e)}")
    
    def check_content(self, 
                     content: str, 
                     denomination: Optional[str] = None,
                     context: Optional[str] = None) -> Dict[str, Any]:
        """
        Check theological content for doctrinal accuracy.
        
        Args:
            content: Text content to check
            denomination: Optional theological tradition to consider
            context: Optional context information (e.g., "Bible_study", "QA")
            
        Returns:
            Dictionary with check results
        """
        results = {
            "passed": True,
            "checks": [],
            "warnings": [],
            "errors": [],
            "score": 100.0,
            "context": context
        }
        
        # Apply rules based on whether denomination is specified
        applied_rules = self._get_applicable_rules(denomination)
        
        # Apply each applicable rule
        for rule_name, rule in applied_rules.items():
            if rule_name == "essential_doctrines":
                continue  # Skip the essential doctrines list
                
            check_result = self._apply_rule(content, rule, rule_name)
            results["checks"].append(check_result)
            
            if check_result["severity"] == "error" and not check_result["passed"]:
                results["passed"] = False
                results["errors"].append(check_result)
                # Major deduction for errors
                results["score"] -= 20.0
            elif check_result["severity"] == "warning" and not check_result["passed"]:
                results["warnings"].append(check_result)
                # Minor deduction for warnings
                results["score"] -= 5.0
        
        # Ensure score doesn't go below 0
        results["score"] = max(0, results["score"])
        
        # Log the results
        if results["passed"]:
            logger.info(f"Theological check passed with score {results['score']}", 
                      denomination=denomination or "general",
                      context=context)
        else:
            logger.warning(f"Theological check failed with score {results['score']}", 
                         denomination=denomination or "general", 
                         context=context,
                         error_count=len(results["errors"]))
            
        return results
    
    def _get_applicable_rules(self, denomination: Optional[str] = None) -> Dict:
        """
        Get rules applicable to the specified denomination or general rules.
        
        Args:
            denomination: Denomination name or None for general rules
            
        Returns:
            Dictionary of applicable rules
        """
        # Start with general rules
        applicable_rules = {}
        if "general" in self.rules:
            applicable_rules.update(self.rules["general"])
        
        # Add denomination-specific rules if specified
        if denomination and denomination in self.rules:
            # Denominational rules override general rules
            applicable_rules.update(self.rules[denomination])
            
        return applicable_rules
            
    def _apply_rule(self, content: str, rule: Dict, rule_name: str) -> Dict:
        """
        Apply a single theological rule to content.
        
        Args:
            content: Text content to check
            rule: Rule definition
            rule_name: Name of the rule
            
        Returns:
            Dictionary with check result
        """
        result = {
            "rule": rule_name,
            "passed": True,
            "description": rule.get("description", ""),
            "severity": rule.get("severity", "warning"),
            "matches": []
        }
        
        # Handle different rule types
        rule_type = rule.get("type", "pattern")
        
        if rule_type == "pattern":
            # Pattern matching rule (regex)
            patterns = rule.get("patterns", [])
            is_forbidden = rule.get("forbidden", True)
            
            for pattern in patterns:
                # Use cached regex pattern if available
                if pattern not in self.pattern_cache:
                    try:
                        self.pattern_cache[pattern] = re.compile(pattern, re.IGNORECASE)
                    except re.error:
                        logger.error(f"Invalid regex pattern: {pattern}")
                        continue
                        
                regex = self.pattern_cache[pattern]
                matches = regex.findall(content)
                
                if matches:
                    for match in matches:
                        result["matches"].append({
                            "pattern": pattern,
                            "match": match if isinstance(match, str) else match[0]
                        })
                    
                    # If forbidden pattern is found, rule fails
                    # If required pattern is not found, rule fails
                    result["passed"] = not is_forbidden
                    break
                    
            # If no matches found for required patterns, rule fails
            if not result["matches"] and not is_forbidden:
                result["passed"] = False
                
        elif rule_type == "verse_required":
            # Check if specific Bible verses are referenced
            required_refs = rule.get("references", [])
            found_refs = set(self._extract_verse_references(content))
            
            missing_refs = []
            for ref in required_refs:
                # Check if any extracted reference contains the required reference
                if not any(ref.lower() in found.lower() for found in found_refs):
                    missing_refs.append(ref)
            
            if missing_refs:
                result["passed"] = False
                result["missing_references"] = missing_refs
        
        elif rule_type == "keyword_sentiment":
            # More complex sentiment analysis around theological keywords
            # This is a placeholder for more advanced implementation
            keywords = rule.get("keywords", [])
            
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    # Simple implementation - could be enhanced with real sentiment analysis
                    context = self._extract_keyword_context(content, keyword)
                    
                    # Check for negative context words
                    negative_context = any(neg in context.lower() 
                                         for neg in rule.get("negative_context", []))
                    
                    if negative_context and not rule.get("allow_negative", False):
                        result["passed"] = False
                        result["matches"].append({
                            "keyword": keyword,
                            "context": context
                        })
        
        return result
    
    def _extract_verse_references(self, content: str) -> List[str]:
        """
        Extract Bible verse references from content.
        
        Args:
            content: Text to extract from
            
        Returns:
            List of verse references
        """
        # Simple regex pattern for verse references
        # This could be enhanced with a more sophisticated verse detection module
        patterns = [
            r'\b(\d?\s?[A-Za-z]+\s\d+:\d+(?:-\d+)?)',  # Genesis 1:1 or Genesis 1:1-5
            r'\b(\d?\s?[A-Za-z]+\s\d+)',               # Genesis 1
        ]
        
        references = []
        for pattern in patterns:
            refs = re.findall(pattern, content)
            references.extend(refs)
            
        return references
    
    def _extract_keyword_context(self, content: str, keyword: str, window: int = 10) -> str:
        """
        Extract surrounding context of a keyword.
        
        Args:
            content: Text content
            keyword: Keyword to find
            window: Number of words around keyword
            
        Returns:
            Context string
        """
        # Find keyword position (case insensitive)
        pos = content.lower().find(keyword.lower())
        if pos == -1:
            return ""
            
        # Split into words and find the index of the word containing the keyword
        words = content.split()
        char_count = 0
        keyword_word_idx = 0
        
        for i, word in enumerate(words):
            if char_count <= pos < char_count + len(word) + 1:
                keyword_word_idx = i
                break
            char_count += len(word) + 1
            
        # Extract context window
        start = max(0, keyword_word_idx - window)
        end = min(len(words), keyword_word_idx + window + 1)
        
        return " ".join(words[start:end])
    
    def check_doctrinal_accuracy(self, 
                               content: str, 
                               doctrine_area: str) -> Dict[str, Any]:
        """
        Check content accuracy for a specific doctrine area.
        
        Args:
            content: Text content to check
            doctrine_area: Doctrine area name (e.g., "Trinity", "Salvation")
            
        Returns:
            Dictionary with check results
        """
        results = {
            "passed": True,
            "doctrine_area": doctrine_area,
            "errors": [],
            "warnings": []
        }
        
        # Get doctrine definition
        doctrine = self.doctrines.get(doctrine_area)
        if not doctrine:
            logger.warning(f"No definition found for doctrine: {doctrine_area}")
            results["warnings"].append(f"Unknown doctrine area: {doctrine_area}")
            return results
        
        # Check required theological terms
        required_terms = doctrine.get("required_terms", [])
        for term in required_terms:
            if term.lower() not in content.lower():
                results["passed"] = False
                results["errors"].append(f"Missing required theological term: {term}")
        
        # Check forbidden theological terms/statements
        forbidden_terms = doctrine.get("forbidden_terms", [])
        for term in forbidden_terms:
            if term.lower() in content.lower():
                results["passed"] = False
                results["errors"].append(f"Contains forbidden theological term: {term}")
        
        # Check if scriptural support is required and present
        if doctrine.get("requires_scripture", False):
            verse_refs = self._extract_verse_references(content)
            if not verse_refs:
                results["passed"] = False
                results["errors"].append("Missing scriptural support")
                
        # Check specific scriptural references if provided
        key_verses = doctrine.get("key_verses", [])
        if key_verses:
            verse_refs = self._extract_verse_references(content)
            found_key_verse = False
            
            for verse in key_verses:
                if any(verse.lower() in ref.lower() for ref in verse_refs):
                    found_key_verse = True
                    break
            
            if not found_key_verse and doctrine.get("requires_key_verse", False):
                results["passed"] = False
                results["warnings"].append("Missing key scriptural reference")
                
        # Log the result
        if results["passed"]:
            logger.info(f"Doctrine check passed for {doctrine_area}")
        else:
            logger.warning(f"Doctrine check failed for {doctrine_area}", 
                         error_count=len(results["errors"]))
            
        return results
    
    def check_denominational_sensitivity(self, 
                                       content: str, 
                                       denomination: str) -> Dict[str, Any]:
        """
        Check if content is sensitive to denominational distinctives.
        
        Args:
            content: Text content to check
            denomination: Denomination name
            
        Returns:
            Dictionary with check results
        """
        results = {
            "passed": True,
            "denomination": denomination,
            "warnings": []
        }
        
        # Get denominational positions
        if denomination not in self.denominational_positions:
            logger.warning(f"No position data for denomination: {denomination}")
            results["warnings"].append(f"Unknown denomination: {denomination}")
            return results
        
        positions = self.denominational_positions[denomination]
        
        # Check distinctive positions