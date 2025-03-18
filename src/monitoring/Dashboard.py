# bible/src/monitoring/dashboard.py
"""
Real-time monitoring dashboard for Bible-AI system using Dash.

Displays metrics like inference latency, theological validation scores, pipeline throughput,
and system resource usage.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests
from typing import Dict, List
from src.utils.logger import get_logger

logger = get_logger("MonitoringDashboard")


class MonitoringDashboard:
    """Real-time dashboard for Bible-AI metrics using Dash."""

    def __init__(self, metrics_port: int = 8000, dashboard_port: int = 8050):
        """
        Initialize the Dash dashboard for monitoring Bible-AI metrics.

        Args:
            metrics_port: Port where Prometheus metrics are exposed.
            dashboard_port: Port to run the Dash server on.
        """
        self.metrics_port = metrics_port
        self.dashboard_port = dashboard_port
        self.app = dash.Dash(__name__)

        # Initialize data storage
        self.data: Dict[str, List] = {
            "timestamp": [],
            "inference_latency": [],
            "validation_score": [],
            "pipeline_throughput": [],
            "cpu_usage": [],
            "memory_usage": [],
        }

        # Define the dashboard layout
        self.app.layout = html.Div(
            [
                html.H1("Bible-AI Monitoring Dashboard", style={"textAlign": "center"}),
                dcc.Interval(id="interval-component", interval=10 * 1000, n_intervals=0),  # Update every 10 seconds
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Inference Latency (seconds)"),
                                dcc.Graph(id="latency-graph"),
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3("Theological Validation Score"),
                                dcc.Graph(id="validation-score-graph"),
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Pipeline Throughput (items/min)"),
                                dcc.Graph(id="throughput-graph"),
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3("System Resources"),
                                dcc.Graph(id="resources-graph"),
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                    ]
                ),
            ]
        )

        # Register callbacks for updating graphs
        self.app.callback(
            [
                Output("latency-graph", "figure"),
                Output("validation-score-graph", "figure"),
                Output("throughput-graph", "figure"),
                Output("resources-graph", "figure"),
            ],
            [Input("interval-component", "n_intervals")],
        )(self.update_graphs)

    def fetch_metrics(self) -> Dict:
        """
        Fetch metrics from the Prometheus endpoint.

        Returns:
            Dictionary of current metric values.
        """
        try:
            response = requests.get(f"http://localhost:{self.metrics_port}/metrics")
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
            logger.error(f"Error fetching metrics: {e}")
            return {}

    def update_graphs(self, n: int) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Update the dashboard graphs with the latest metrics.

        Args:
            n: Number of intervals (from dcc.Interval).

        Returns:
            Tuple of Plotly figures for each graph.
        """
        metrics = self.fetch_metrics()

        # Update data storage
        timestamp = pd.Timestamp.now()
        self.data["timestamp"].append(timestamp)
        self.data["inference_latency"].append(
            metrics.get("bibleai_inference_latency_seconds_sum", 0)
            / max(1, metrics.get("bibleai_inference_latency_seconds_count", 1))
        )
        self.data["validation_score"].append(
            metrics.get("bibleai_theological_validation_score", 0)
        )
        self.data["pipeline_throughput"].append(
            metrics.get("bibleai_pipeline_throughput", 0)
        )
        self.data["cpu_usage"].append(metrics.get("bibleai_cpu_usage_percent", 0))
        self.data["memory_usage"].append(metrics.get("bibleai_memory_usage_mb", 0))

        # Keep only the last 60 data points (10 minutes at 10-second intervals)
        for key in self.data:
            self.data[key] = self.data[key][-60:]

        # Create DataFrame
        df = pd.DataFrame(self.data)

        # Create graphs
        latency_fig = px.line(
            df, x="timestamp", y="inference_latency", title="Inference Latency Over Time"
        )
        validation_fig = px.line(
            df, x="timestamp", y="validation_score", title="Theological Validation Score Over Time"
        )
        throughput_fig = px.line(
            df, x="timestamp", y="pipeline_throughput", title="Pipeline Throughput Over Time"
        )
        resources_fig = px.line(
            df,
            x="timestamp",
            y=["cpu_usage", "memory_usage"],
            title="System Resources Over Time",
            labels={"value": "Usage", "variable": "Metric"},
        )

        return latency_fig, validation_fig, throughput_fig, resources_fig

    def run(self) -> None:
        """Run the Dash server."""
        try:
            self.app.run_server(debug=False, host="0.0.0.0", port=self.dashboard_port)
            logger.info(f"Dashboard running on http://localhost:{self.dashboard_port}")
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            raise


if __name__ == "__main__":
    dashboard = MonitoringDashboard(metrics_port=8000, dashboard_port=8050)
    dashboard.run()
