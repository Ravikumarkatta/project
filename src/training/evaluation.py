# src/training/evaluation.py
"""Advanced evaluation module for Bible-AI with theological and multi-task metrics."""
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Callable, Optional, Union, Any
from collections import defaultdict
import logging
import os
from threading import Thread
from queue import Queue
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Project-specific imports
from src.model.architecture import BiblicalTransformer
from src.data.dataset import BibleDataset
from src.theology.validator import TheologicalValidator
from src.utils.logger import setup_logger

# Project root for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    import sys
    sys.path.insert(0, PROJECT_ROOT)

# Setup logging
log_dir = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(log_dir, exist_ok=True)
logger = setup_logger("evaluation", os.path.join(log_dir, "evaluation.log"))

@dataclass
class MetricResult:
    """Container for metric results with serialization support."""
    name: str
    value: Union[float, Dict[str, float]]
    step: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "step": self.step}

class BibleEvaluator:
    """Evaluation system for biblical text generation models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        config: Dict,
        theological_validator: Optional[TheologicalValidator] = None
    ):
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.theological_validator = theological_validator or TheologicalValidator()
        self.metrics = defaultdict(float)
        self.device = next(model.parameters()).device
    
    def compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        verse_logits: Optional[torch.Tensor] = None,
        verse_labels: Optional[torch.Tensor] = None,
        theological_logits: Optional[torch.Tensor] = None,
        theological_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            logits: Main model logits
            labels: Ground truth labels
            verse_logits: Verse detection logits (optional)
            verse_labels: Verse labels (optional)
            theological_logits: Theological classification logits (optional)
            theological_labels: Theological labels (optional)
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Main task metrics (text generation)
        pred_tokens = torch.argmax(logits, dim=-1)
        valid_mask = (labels != -100)  # Ignore padding
        
        # Accuracy
        accuracy = (pred_tokens[valid_mask] == labels[valid_mask]).float().mean().item()
        metrics['accuracy'] = accuracy
        
        # Perplexity
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        metrics['perplexity'] = torch.exp(loss).item()
        
        # Verse detection metrics if provided
        if verse_logits is not None and verse_labels is not None:
            verse_preds = torch.argmax(verse_logits, dim=-1)
            verse_mask = (verse_labels != -100)
            
            verse_accuracy = (verse_preds[verse_mask] == verse_labels[verse_mask]).float().mean().item()
            metrics['verse_detection_accuracy'] = verse_accuracy
            
            # Calculate precision, recall, F1 for verse detection
            y_true = verse_labels[verse_mask].cpu().numpy()
            y_pred = verse_preds[verse_mask].cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average='weighted'
            )
            metrics.update({
                'verse_detection_precision': precision,
                'verse_detection_recall': recall,
                'verse_detection_f1': f1
            })
        
        # Theological accuracy metrics if provided
        if theological_logits is not None and theological_labels is not None:
            theo_preds = torch.argmax(theological_logits, dim=-1)
            theo_truth = torch.argmax(theological_labels, dim=-1)
            
            theo_accuracy = (theo_preds == theo_truth).float().mean().item()
            metrics['theological_accuracy'] = theo_accuracy
            
            # Calculate theological metrics using validator
            if hasattr(self.model, 'tokenizer'):
                pred_text = self.model.tokenizer.batch_decode(pred_tokens)
                validation_scores = [
                    self.theological_validator.validate({"text": text})
                    for text in pred_text
                ]
                metrics['theological_validation_score'] = np.mean(validation_scores)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> Dict[str, float]:
        """
        Run evaluation loop over validation data.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary of averaged metrics
        """
        self.model.eval()
        total_metrics = defaultdict(float)
        num_batches = 0
        
        for batch in self.val_loader:
            # Move all batch tensors to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Compute metrics
            batch_metrics = self.compute_metrics(
                outputs['logits'],
                batch['labels'],
                outputs.get('verse_logits'),
                batch.get('verse_labels'),
                outputs.get('theological_logits'),
                batch.get('theological_labels')
            )
            
            # Accumulate metrics
            for metric, value in batch_metrics.items():
                total_metrics[metric] += value
            num_batches += 1
        
        # Average metrics
        metrics = {
            metric: value / num_batches
            for metric, value in total_metrics.items()
        }
        
        # Log metrics
        metrics_str = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        print(f"Epoch {epoch} Validation Metrics:")
        print(" | ".join(metrics_str))
        
        return metrics

if __name__ == "__main__":
    # Example usage
    with open(os.path.join(PROJECT_ROOT, "config/training_config.json"), "r") as f:
        config = json.load(f)
    
    model = BiblicalTransformer(BiblicalTransformerConfig(**config.get("model_params", {})))
    val_data = BibleDataset("val", config.get("data", {}))  # Placeholder
    val_loader = DataLoader(val_data, batch_size=config.get("training", {}).get("batch_size", 16))
    
    evaluator = BibleEvaluator(model, val_loader, config)
    results = evaluator.evaluate(epoch=1)
    print(results)
