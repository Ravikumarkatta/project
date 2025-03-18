# bible/src/monitoring/metrics.py
"""
Metrics collection for Bible-AI system.

This module collects metrics on model performance, data pipeline health, and theological validation.
Metrics are exposed via Prometheus for monitoring and alerting.
"""

import time
from typing import Dict, Optional, List
import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from src.utils.logger import get_logger
from src.theology.validator import TheologicalValidator

logger = get_logger("MetricsCollector")


class MetricsCollector:
    """Collects and exposes metrics for Bible-AI system monitoring."""

    def __init__(self, port: int = 8000, validator_config: Optional[Dict] = None):
        """
        Initialize the metrics collector with Prometheus metrics.

        Args:
            port: Port to expose metrics on (for Prometheus scraping).
            validator_config: Configuration for TheologicalValidator.
        """
        self.inference_counter = Counter(
            "bibleai_inference_total", "Total number of inference requests processed"
        )
        self.inference_latency = Histogram(
            "bibleai_inference_latency_seconds",
            "Latency of inference requests in seconds",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")],
        )
        self.validation_score = Gauge(
            "bibleai_theological_validation_score",
            "Theological validation score of responses (0 to 1)",
        )
        self.pipeline_throughput = Gauge(
            "bibleai_pipeline_throughput",
            "Number of data items processed per minute in the pipeline",
        )
        self.cpu_usage = Gauge("bibleai_cpu_usage_percent", "CPU usage percentage")
        self.memory_usage = Gauge("bibleai_memory_usage_mb", "Memory usage in MB")

        # Start the Prometheus HTTP server to expose metrics
        try:
            start_http_server(port)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

        # Initialize TheologicalValidator if config is provided
        self.validator = (
            TheologicalValidator(validator_config) if validator_config else None
        )

    def track_inference(self, latency: float) -> None:
        """
        Track an inference request's latency and increment the counter.

        Args:
            latency: Time taken for inference in seconds.
        """
        try:
            self.inference_counter.inc()
            self.inference_latency.observe(latency)
            logger.debug(f"Tracked inference with latency {latency:.2f}s")
        except Exception as e:
            logger.error(f"Error tracking inference metrics: {e}")

    def track_validation_score(self, response: Dict) -> None:
        """
        Track the theological validation score of a response.

        Args:
            response: Response dictionary to validate.
        """
        if not self.validator:
            logger.warning("Theological validator not initialized; skipping validation score")
            return
        try:
            score = self.validator.validate(response)
            self.validation_score.set(score)
            logger.debug(f"Theological validation score: {score:.2f}")
        except Exception as e:
            logger.error(f"Error tracking validation score: {e}")

    def track_pipeline_throughput(self, items_processed: int, duration: float) -> None:
        """
        Track the throughput of the data pipeline (items per minute).

        Args:
            items_processed: Number of items processed.
            duration: Time taken in seconds.
        """
        try:
            if duration <= 0:
                raise ValueError("Duration must be positive")
            throughput = (items_processed / duration) * 60  # Convert to items per minute
            self.pipeline_throughput.set(throughput)
            logger.debug(f"Pipeline throughput: {throughput:.2f} items/min")
        except Exception as e:
            logger.error(f"Error tracking pipeline throughput: {e}")

    def track_system_resources(self) -> None:
        """Track CPU and memory usage of the system."""
        try:
            self.cpu_usage.set(psutil.cpu_percent(interval=1))
            self.memory_usage.set(psutil.virtual_memory().used / (1024 * 1024))  # Convert to MB
            logger.debug(
                f"System resources - CPU: {self.cpu_usage._value.get():.2f}%, "
                f"Memory: {self.memory_usage._value.get():.2f}MB"
            )
        except Exception as e:
            logger.error(f"Error tracking system resources: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize metrics collector
    validator_config = {"min_score": 0.9, "theological_terms": ["God", "Jesus", "Holy Spirit"]}
    collector = MetricsCollector(port=8000, validator_config=validator_config)

    # Simulate tracking metrics
    collector.track_inference(0.5)  # Simulate 0.5s inference
    collector.track_validation_score({"text": "Jesus said, 'Love one another.'"})
    collector.track_pipeline_throughput(100, 60)  # 100 items in 60 seconds
    collector.track_system_resources()

    # Keep the script running to expose metrics
    import time

    while True:
        time.sleep(60)  # Update system resources every minute
        collector.track_system_resources()
