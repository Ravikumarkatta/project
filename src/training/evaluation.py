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

class EvaluationRegistry:
    """Dynamic registry for evaluation metrics."""
    def __init__(self):
        self.metrics: Dict[str, Callable] = {}
    
    def register(self, name: str, metric_fn: Callable) -> None:
        """Register a new metric function."""
        if not callable(metric_fn):
            raise ValueError(f"Metric {name} must be callable")
        self.metrics[name] = metric_fn
        logger.info(f"Registered metric: {name}")
    
    def get(self, name: str) -> Callable:
        """Retrieve a metric function by name."""
        return self.metrics.get(name, lambda *args, **kwargs: None)

class BibleEvaluator:
    """Robust evaluator for Bible-AI with theological and multi-task evaluation."""
    
    def __init__(
        self,
        model: BiblicalTransformer,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device = None,
        theological_validator: Optional[TheologicalValidator] = None
    ):
        """Initialize evaluator with model, data, and config."""
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theological_validator = theological_validator or TheologicalValidator(config.get("theology", {}))
        self.writer = SummaryWriter(os.path.join(log_dir, "tensorboard_eval"))
        self.registry = EvaluationRegistry()
        self.global_step = 0
        
        # Register default metrics
        self._register_default_metrics()
        
        # Initialize queues for parallel evaluation
        self.result_queue = Queue()
        self.model.eval().to(self.device)
        logger.info("Evaluator initialized with device: %s", self.device)

    def _register_default_metrics(self):
        """Register built-in metrics for Bible-AI."""
        self.registry.register("loss", self._compute_loss)
        self.registry.register("accuracy", self._compute_accuracy)
        self.registry.register("precision_recall_f1", self._compute_precision_recall_f1)
        self.registry.register("theological_alignment", self._compute_theological_alignment)

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> float:
        """Compute average loss."""
        from src.training.loss import TheologicalLoss  # Lazy import to avoid circular dependency
        criterion = TheologicalLoss(**self.config.get("loss", {}))
        return criterion(outputs["logits"], targets).item()

    def _compute_accuracy(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> float:
        """Compute accuracy."""
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        targets = targets.cpu().numpy()
        return accuracy_score(targets, preds)

    def _compute_precision_recall_f1(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """Compute precision, recall, and F1 scores."""
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        targets = targets.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average="weighted", zero_division=0)
        return {"precision": precision, "recall": recall, "f1": f1}

    def _compute_theological_alignment(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> float:
        """Compute a custom Theological Alignment Score (TAS)."""
        preds = torch.argmax(outputs["logits"], dim=-1)
        score = self.theological_validator.validate(outputs["logits"], targets)
        return score  # Assumes validator returns a float between 0 and 1

    def add_custom_metric(self, name: str, metric_fn: Callable) -> None:
        """Add a custom metric to the registry."""
        self.registry.register(name, metric_fn)

    def _evaluate_batch(self, batch: Dict[str, torch.Tensor], metric_names: List[str]) -> Dict[str, float]:
        """Evaluate a single batch with specified metrics."""
        try:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["labels"].to(self.device)
            
            with torch.no_grad():
                if self.config.get("training", {}).get("mixed_precision", False):
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            results = {}
            for name in metric_names:
                metric_fn = self.registry.get(name)
                result = metric_fn(outputs, targets)
                results[name] = result
            return results
        except Exception as e:
            logger.error("Batch evaluation failed: %s", e)
            return {}

    def _parallel_evaluate(self, metric_names: List[str], result_queue: Queue) -> None:
        """Evaluate dataset in parallel threads."""
        batch_results = []
        for batch in self.val_loader:
            result = self._evaluate_batch(batch, metric_names)
            batch_results.append(result)
        result_queue.put(batch_results)

    def evaluate(
        self,
        metric_names: Optional[List[str]] = None,
        epoch: int = 0,
        visualize: bool = True
    ) -> Dict[str, float]:
        """Perform full evaluation with parallel processing and visualization."""
        metric_names = metric_names or list(self.registry.metrics.keys())
        if not metric_names:
            logger.warning("No metrics specified for evaluation")
            return {}
        
        logger.info("Starting evaluation with metrics: %s", metric_names)
        
        # Start parallel evaluation
        num_threads = self.config.get("evaluation", {}).get("num_threads", 2)
        batch_splits = np.array_split(list(self.val_loader), num_threads)
        threads = []
        queues = [Queue() for _ in range(num_threads)]
        
        for i, split in enumerate(batch_splits):
            loader = DataLoader(
                self.val_loader.dataset,
                batch_size=self.val_loader.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(range(len(split) * self.val_loader.batch_size))
            )
            thread = Thread(target=self._parallel_evaluate, args=(metric_names, queues[i]))
            thread.start()
            threads.append(thread)
        
        # Collect results
        all_results = defaultdict(list)
        for queue in queues:
            batch_results = queue.get()
            for result in batch_results:
                for name, value in result.items():
                    all_results[name].append(value)
        
        for thread in threads:
            thread.join()
        
        # Aggregate results
        aggregated_results = {}
        for name, values in all_results.items():
            if isinstance(values[0], dict):
                aggregated_results[name] = {
                    k: np.mean([v[k] for v in values]) for k in values[0].keys()
                }
            else:
                aggregated_results[name] = np.mean(values)
        
        # Log and visualize
        for name, value in aggregated_results.items():
            self.writer.add_scalar(f"Eval/{name}", value if isinstance(value, float) else value["f1"], epoch)
            logger.info(f"Epoch {epoch} - {name}: {value}")
            self.global_step += 1
            self.result_queue.put(MetricResult(name, value, self.global_step))
        
        if visualize:
            self._visualize_metrics(aggregated_results, epoch)
        
        return aggregated_results

    def _visualize_metrics(self, metrics: Dict[str, Union[float, Dict[str, float]]], epoch: int) -> None:
        """Generate custom visualization of metrics."""
        plt.figure(figsize=(10, 6))
        for name, value in metrics.items():
            if isinstance(value, dict):
                plt.plot(epoch, value["f1"], label=f"{name}_f1", marker="o")
            else:
                plt.plot(epoch, value, label=name, marker="o")
        plt.title(f"Evaluation Metrics - Epoch {epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(log_dir, f"metrics_epoch_{epoch}.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved metric visualization at {plot_path}")
        self.writer.add_image("Metrics Plot", plt.imread(plot_path), epoch, dataformats="HWC")

    def save_results(self, output_path: str) -> None:
        """Save evaluation results to disk."""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get().to_dict())
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved evaluation results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.writer.close()
        torch.cuda.empty_cache()
        logger.info("Evaluator cleanup completed")

def evaluate_model(
    model: BiblicalTransformer,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    metric_names: Optional[List[str]] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """Convenience function for one-off evaluation."""
    evaluator = BibleEvaluator(model, val_loader, config, device)
    results = evaluator.evaluate(metric_names, epoch)
    evaluator.cleanup()
    return results

if __name__ == "__main__":
    # Example usage
    with open(os.path.join(PROJECT_ROOT, "config/training_config.json"), "r") as f:
        config = json.load(f)
    
    model = BiblicalTransformer(BiblicalTransformerConfig(**config.get("model_params", {})))
    val_data = BibleDataset("val", config.get("data", {}))  # Placeholder
    val_loader = DataLoader(val_data, batch_size=config.get("training", {}).get("batch_size", 16))
    
    evaluator = BibleEvaluator(model, val_loader, config)
    results = evaluator.evaluate(epoch=1)
    evaluator.save_results(os.path.join(PROJECT_ROOT, "results/eval_results.json"))
    print(results)
