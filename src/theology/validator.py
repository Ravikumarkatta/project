import json
import re
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class TheologicalValidator:
    """System for validating theological accuracy of model outputs."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize theological rules
        self.rules = self._initialize_rules()
        
        # Load Bible knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
        # Setup embeddings model for semantic similarity
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load theological validation configuration."""
        if not config_path:
            config_path = Path(__file__).parent.parent.parent / "config" / "theological_rules.json"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {
                "min_doctrinal_score": 0.7,
                "min_biblical_accuracy": 0.8,
                "semantic_similarity_threshold": 0.85,
                "essential_doctrines": [
                    "trinity",
                    "deity_of_christ",
                    "salvation_by_grace",
                    "resurrection",
                    "scripture_authority"
                ],
                "heretical_patterns": [
                    "works_based_salvation",
                    "denial_of_trinity",
                    "denial_of_christ_deity"
                ]
            }
    
    def _initialize_rules(self) -> Dict:
        """Initialize theological validation rules."""
        rules_path = Path(__file__).parent / "rules"
        rules = {}
        
        # Core doctrinal rules
        rules["doctrinal"] = {
            doctrine: self._load_rule(rules_path / f"{doctrine}.json")
            for doctrine in self.config["essential_doctrines"]
        }
        
        # Heresy detection rules
        rules["heretical"] = {
            pattern: self._load_rule(rules_path / f"{pattern}.json")
            for pattern in self.config["heretical_patterns"]
        }
        
        return rules
    
    def _load_rule(self, rule_path: Path) -> Dict:
        """Load individual validation rule."""
        try:
            if rule_path.exists():
                with open(rule_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Rule file not found: {rule_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load rule {rule_path}: {e}")
            return {}
    
    def _load_knowledge_base(self) -> Dict:
        """Load biblical knowledge base for validation."""
        kb_path = Path(__file__).parent / "knowledge_base" / "theological_kb.json"
        try:
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning("Knowledge base file not found")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            return {}
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text using sentence transformer."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling to get text embedding
            embedding = torch.mean(outputs.last_hidden_state, dim=1)
        
        return embedding
    
    def _check_doctrinal_accuracy(self, text: str) -> Dict[str, float]:
        """Check text against core doctrinal rules."""
        scores = {}
        text_embedding = self._get_text_embedding(text)
        
        for doctrine, rule in self.rules["doctrinal"].items():
            if "key_statements" in rule:
                # Compare with key doctrinal statements
                statement_scores = []
                for statement in rule["key_statements"]:
                    statement_embedding = self._get_text_embedding(statement)
                    similarity = torch.cosine_similarity(text_embedding, statement_embedding)
                    statement_scores.append(similarity.item())
                scores[doctrine] = max(statement_scores)
            
            if "keywords" in rule:
                # Check for presence of essential keywords
                keyword_score = sum(1 for word in rule["keywords"] if word.lower() in text.lower())
                scores[f"{doctrine}_keywords"] = keyword_score / len(rule["keywords"])
        
        return scores
    
    def _check_heretical_patterns(self, text: str) -> Dict[str, bool]:
        """Check text for heretical patterns."""
        results = {}
        for pattern, rule in self.rules["heretical"].items():
            if "patterns" in rule:
                matches = any(re.search(p, text, re.IGNORECASE) for p in rule["patterns"])
                results[pattern] = matches
        return results
    
    def validate(self, text_data: Union[str, Dict]) -> float:
        """
        Validate theological accuracy of text.
        
        Args:
            text_data: Either a string of text or a dictionary containing text and metadata
            
        Returns:
            Validation score between 0 and 1
        """
        if isinstance(text_data, dict):
            text = text_data.get("text", "")
            context = text_data.get("context", {})
        else:
            text = text_data
            context = {}
        
        if not text:
            return 0.0
        
        # Check doctrinal accuracy
        doctrinal_scores = self._check_doctrinal_accuracy(text)
        avg_doctrinal_score = sum(doctrinal_scores.values()) / len(doctrinal_scores)
        
        # Check for heretical patterns
        heresy_checks = self._check_heretical_patterns(text)
        heresy_penalty = sum(1 for check in heresy_checks.values() if check) * 0.2
        
        # Calculate final score
        base_score = avg_doctrinal_score
        final_score = max(0.0, base_score - heresy_penalty)
        
        # Log validation results
        self.logger.info(f"Validation results for text: {final_score:.2f}")
        self.logger.debug(f"Doctrinal scores: {doctrinal_scores}")
        self.logger.debug(f"Heresy checks: {heresy_checks}")
        
        return final_score
    
    def validate_batch(self, texts: List[Union[str, Dict]]) -> List[float]:
        """Validate a batch of texts."""
        return [self.validate(text) for text in texts]