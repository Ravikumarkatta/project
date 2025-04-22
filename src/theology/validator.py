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
                        
                        # Process orthodox statements as doctrinal
                        for stmt in data.get("orthodox_statements", []):
                            # Add to a generic category if no specific category is provided
                            category = "general"
                            if category not in rules["doctrinal"]:
                                rules["doctrinal"][category] = {
                                    "key_statements": [],
                                    "keywords": []
                                }
                            rules["doctrinal"][category]["key_statements"].append(stmt)
                        
                        # Process heterodox statements as heretical
                        for stmt in data.get("heterodox_statements", []):
                            # Add to a generic category if no specific category is provided
                            category = "general"
                            if category not in rules["heretical"]:
                                rules["heretical"][category] = {
                                    "patterns": []
                                }
                            # Convert heterodox statements to regex patterns
                            pattern = r".*" + re.escape(stmt) + r".*"
                            rules["heretical"][category]["patterns"].append(pattern)
                            
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
        if not statement.strip():
            return 0.5  # Neutral score for empty text
            
        try:
            # Get statement embedding
            statement_embedding = self.model.encode([statement])[0]
            
            # Collect all doctrinal statements
            orthodox_statements = []
            for category, content in self.rules["doctrinal"].items():
                orthodox_statements.extend(content.get("key_statements", []))
            
            # Calculate similarity with orthodox statements
            if orthodox_statements:
                orthodox_embeddings = self.model.encode(orthodox_statements)
                orthodox_sims = cosine_similarity(
                    [statement_embedding], orthodox_embeddings
                )[0]
                max_orthodox_sim = np.max(orthodox_sims)
            else:
                max_orthodox_sim = 0
                
            # Check for heretical patterns
            heretical_match = False
            for category, content in self.rules["heretical"].items():
                for pattern in content.get("patterns", []):
                    if re.search(pattern, statement, re.IGNORECASE):
                        heretical_match = True
                        break
                if heretical_match:
                    break
            
            # Adjust score based on heretical match
            max_heterodox_sim = 0.8 if heretical_match else 0
                
            # Calculate final score
            # Higher similarity to orthodox and lower to heterodox statements results in higher score
            score = (max_orthodox_sim - max_heterodox_sim + 1) / 2
            
            return float(min(max(score, 0.0), 1.0))  # Ensure score is between 0 and 1

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
