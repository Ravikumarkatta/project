"""Theological validator module."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TheologicalValidator:
    """Validator for theological statements."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize validator with embedding model."""
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> Dict[str, Any]:
        """Load theological rules from JSON files."""
        rules_dir = Path(__file__).parent / "rules"
        rules = {
            "doctrinal": {},
            "heretical": {}
        }

        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file) as f:
                        data = json.load(f)
                        
                        # Process orthodox statements
                        if "orthodox_statements" in data:
                            if "general" not in rules["doctrinal"]:
                                rules["doctrinal"]["general"] = {
                                    "key_statements": [],
                                    "keywords": []
                                }
                            rules["doctrinal"]["general"]["key_statements"].extend(
                                data.get("orthodox_statements", [])
                            )
                        
                        # Process heterodox statements
                        if "heterodox_statements" in data:
                            if "general" not in rules["heretical"]:
                                rules["heretical"]["general"] = {
                                    "patterns": []
                                }
                            # Convert heterodox statements to regex patterns
                            for stmt in data.get("heterodox_statements", []):
                                pattern = re.escape(stmt).replace(r"\ ", r"\s+")
                                rules["heretical"]["general"]["patterns"].append(pattern)
                        
                        # Directly process doctrinal and heretical sections if they exist
                        if "doctrinal" in data:
                            for category, content in data["doctrinal"].items():
                                rules["doctrinal"][category] = content
                                
                        if "heretical" in data:
                            for category, content in data["heretical"].items():
                                rules["heretical"][category] = content
                            
                except Exception as e:
                    self.logger.error(f"Error loading rules from {rule_file}: {e}")

        return rules

    def validate(self, statement: str) -> float:
        """Validate a theological statement.

        Args:
            statement: The statement to validate

        Returns:
            Validation score between 0 and 1
        """
        if not statement or not statement.strip():
            return 0.5  # Neutral score for empty text
            
        try:
            # Track scores
            orthodox_score = 0.0
            heterodox_score = 0.0
            
            # Check doctrinal statements with embeddings
            all_key_statements = []
            for category, content in self.rules["doctrinal"].items():
                key_statements = content.get("key_statements", [])
                if key_statements:
                    all_key_statements.extend(key_statements)
                    
                # Also check keywords
                for keyword in content.get("keywords", []):
                    if re.search(r'\b' + re.escape(keyword) + r'\b', statement, re.IGNORECASE):
                        orthodox_score += 0.1  # Small boost for keyword match
            
            # Get embeddings for orthodox statements
            if all_key_statements:
                statement_embedding = self.model.encode([statement])[0]
                orthodox_embeddings = self.model.encode(all_key_statements)
                orthodox_sims = cosine_similarity(
                    [statement_embedding], orthodox_embeddings
                )[0]
                
                # Update orthodox score with max similarity
                if len(orthodox_sims) > 0:
                    orthodox_score = max(orthodox_score, float(np.max(orthodox_sims)))
            
            # Check heretical patterns
            for category, content in self.rules["heretical"].items():
                for pattern in content.get("patterns", []):
                    if re.search(pattern, statement, re.IGNORECASE):
                        heterodox_score = max(heterodox_score, 0.8)  # Strong penalty for matching heretical pattern
                        break
            
            # Calculate final score (ensure it's between 0 and 1)
            # Higher orthodox score and lower heterodox score results in higher final score
            score = (orthodox_score - heterodox_score + 1) / 2
            score = min(max(score, 0.0), 1.0)
            
            return float(score)

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return 0.5  # Return neutral score on error

    def validate_batch(self, statements: List[str]) -> List[float]:
        """Validate multiple statements.

        Args:
            statements: List of statements to validate

        Returns:
            List of validation scores
        """
        return [self.validate(stmt) for stmt in statements]
