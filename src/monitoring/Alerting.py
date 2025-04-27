"""
Alerting system for Bible-AI monitoring.

Monitors metrics and sends alerts via email (and optionally Slack) when thresholds are exceeded.
"""

import smtplib
import time
from email.mime.text import MIMEText
from typing import Dict, Optional

import requests

from src.utils.logger import get_logger

logger = get_logger("AlertingSystem")


class AlertingSystem:
    """Monitors metrics and sends alerts when thresholds are exceeded."""

    def __init__(
        self,
        metrics_port: int = 8000,
        smtp_config: Optional[Dict[str, str]] = None,
        slack_webhook: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the alerting system.

        Args:
            metrics_port: Port where Prometheus metrics are exposed.
            smtp_config: Dictionary with SMTP server configuration (host, port, user, password, to_email).
            slack_webhook: Slack webhook URL for notifications (optional).
            thresholds: Dictionary of metric thresholds for alerts.
        """
        self.metrics_port = metrics_port
        self.smtp_config = smtp_config or {}
        self.slack_webhook = slack_webhook
        self.thresholds = thresholds or {
            "inference_latency": 2.0,  # Alert if average latency > 2 seconds
            "validation_score": 0.9,  # Alert if score < 0.9
            "cpu_usage": 80.0,  # Alert if CPU usage > 80%
            "memory_usage": 5000.0,  # Alert if memory usage > 5000 MB
        }
        self.alert_cooldown = 300  # 5 minutes cooldown between alerts
        self.last_alerted: Dict[str, float] = {}

    def fetch_metrics(self) -> Dict[str, float]:
        """
        Fetch metrics from the Prometheus endpoint.

        Returns:
            Dictionary of current metric values.
        """
        try:
            response = requests.get(
                f"http://localhost:{self.metrics_port}/metrics", timeout=5
            )
            response.raise_for_status()
            lines = response.text.splitlines()
            metrics = {}
            for line in lines:
                if line.startswith("bibleai_"):
                    parts = line.split()
                    metric_name = parts[0].split("{")[0]
                    value = float(parts[-1])
                    metrics[metric_name] = value
            return metrics
        except Exception as e:
            logger.error(f"Error fetching metrics for alerting: {e}")
            return {}

    def send_email_alert(self, subject: str, message: str) -> None:
        """
        Send an email alert using SMTP.

        Args:
            subject: Email subject.
            message: Email body.
        """
        if not self.smtp_config:
            logger.warning("SMTP config not provided; skipping email alert")
            return

        smtp_host = self.smtp_config.get("host", "")
        smtp_port = int(self.smtp_config.get("port", 587))
        smtp_user = self.smtp_config.get("user", "")
        smtp_password = self.smtp_config.get("password", "")
        to_email = self.smtp_config.get("to_email", "")

        if not all([smtp_host, smtp_user, smtp_password, to_email]):
            logger.error("Incomplete SMTP configuration; skipping email alert")
            return

        for attempt in range(3):  # Retry up to 3 times
            try:
                msg = MIMEText(message)
                msg["Subject"] = subject
                msg["From"] = smtp_user
                msg["To"] = to_email

                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                    server.sendmail(
                        smtp_user,
                        [to_email],  # list of recipients
                        msg.as_string(),
                    )
                logger.info(f"Sent email alert: {subject}")
                return
            except Exception as e:
                logger.error(f"Attempt {attempt+1}: Error sending email alert: {e}")
                time.sleep(2)  # Wait 2 seconds before retrying
        logger.error("Failed to send email alert after 3 attempts.")

    def send_slack_alert(self, message: str) -> None:
        """
        Send a Slack alert using a webhook.

        Args:
            message: Message to send.
        """
        if not self.slack_webhook:
            logger.warning("Slack webhook not provided; skipping Slack alert")
            return
        try:
            payload = {"text": message}
            response = requests.post(self.slack_webhook, json=payload, timeout=5)
            response.raise_for_status()
            logger.info("Sent Slack alert")
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    def check_alerts(self) -> None:
        """Check metrics against thresholds and send alerts if necessary."""
        metrics = self.fetch_metrics()
        current_time = time.time()

        latency_sum = metrics.get("bibleai_inference_latency_seconds_sum", 0.0)
        latency_count = metrics.get("bibleai_inference_latency_seconds_count", 1.0)
        avg_latency = latency_sum / max(1.0, latency_count)

        checks = [
            (
                "inference_latency",
                avg_latency,
                self.thresholds["inference_latency"],
                "gt",
            ),
            (
                "validation_score",
                metrics.get("bibleai_theological_validation_score", 1.0),
                self.thresholds["validation_score"],
                "lt",
            ),
            (
                "cpu_usage",
                metrics.get("bibleai_cpu_usage_percent", 0.0),
                self.thresholds["cpu_usage"],
                "gt",
            ),
            (
                "memory_usage",
                metrics.get("bibleai_memory_usage_mb", 0.0),
                self.thresholds["memory_usage"],
                "gt",
            ),
        ]

        for metric_name, value, threshold, comparison in checks:
            should_alert = (comparison == "gt" and value > threshold) or (
                comparison == "lt" and value < threshold
            )
            last_alert_time = self.last_alerted.get(metric_name, 0)

            if should_alert and (current_time - last_alert_time) > self.alert_cooldown:
                message = f"Alert: {metric_name} exceeded threshold. Value: {value:.2f}, Threshold: {threshold}"
                logger.warning(message)

                # Send alerts
                self.send_email_alert(f"Bible-AI Alert: {metric_name}", message)
                self.send_slack_alert(message)

                self.last_alerted[metric_name] = current_time


if __name__ == "__main__":
    smtp_config = {
        "host": "smtp.example.com",
        "port": "587",
        "user": "your-email@example.com",
        "password": "your-password",
        "to_email": "alert-recipient@example.com",
    }
    slack_webhook = "https://hooks.slack.com/services/your/webhook/url"

    alerting = AlertingSystem(
        metrics_port=8000,
        smtp_config=smtp_config,
        slack_webhook=slack_webhook,
    )

    while True:
        alerting.check_alerts()
        time.sleep(60)  # Check every minute
