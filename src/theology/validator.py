"""Theological validator module."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TheologicalValidator:
    """Validator for theological statements."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize validator with embedding model."""
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, List[str]]:
        """Load theological rules from JSON files."""
        rules_dir = Path(__file__).parent / "rules"
        rules = {"orthodox": [], "heterodox": []}

        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file) as f:
                        data = json.load(f)
                        rules["orthodox"].extend(data.get("orthodox_statements", []))
                        rules["heterodox"].extend(data.get("heterodox_statements", []))
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
        try:
            # Get statement embedding
            statement_embedding = self.model.encode([statement])[0]

            # Get embeddings for orthodox and heterodox statements
            orthodox_embeddings = self.model.encode(self.rules["orthodox"])
            heterodox_embeddings = self.model.encode(self.rules["heterodox"])

            # Calculate similarities
            orthodox_sims = cosine_similarity(
                [statement_embedding], orthodox_embeddings
            )[0]
            heterodox_sims = cosine_similarity(
                [statement_embedding], heterodox_embeddings
            )[0]

            # Get max similarities
            max_orthodox_sim = np.max(orthodox_sims) if len(orthodox_sims) > 0 else 0
            max_heterodox_sim = np.max(heterodox_sims) if len(heterodox_sims) > 0 else 0

            # Calculate final score
            # Higher similarity to orthodox and lower to heterodox statements results in higher score
            score = (max_orthodox_sim - max_heterodox_sim + 1) / 2

            return float(score)

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return 0.0

    def validate_batch(self, statements: List[str]) -> List[float]:
        """Validate multiple statements.

        Args:
            statements: List of statements to validate

        Returns:
            List of validation scores
        """
        return [self.validate(stmt) for stmt in statements]
