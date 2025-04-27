#!/usr/bin/env python3
"""Health check script for Bible-AI components."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import psutil
import requests
from rich.console import Console
from rich.table import Table

console = Console()


def check_api_health(url: str, timeout: int = 5) -> Dict[str, bool]:
    """Check health of API endpoints."""
    endpoints = {
        "health": "/health",
        "api": "/api/v1/status",
        "model": "/api/v1/model/status",
        "theology": "/api/v1/theology/status",
    }

    results = {}
    for name, path in endpoints.items():
        try:
            response = requests.get(f"{url}{path}", timeout=timeout)
            results[name] = response.status_code == 200
        except requests.RequestException:
            results[name] = False

    return results


def check_system_health() -> Dict[str, float]:
    """Check system resource usage."""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }


def check_model_files() -> Dict[str, bool]:
    """Check if required model files exist."""
    required_files = [
        "data/embeddings",
        "config/model_config.json",
        "config/theological_rules.json",
    ]

    return {path: Path(path).exists() for path in required_files}


def check_logs() -> Dict[str, Optional[str]]:
    """Check log files for errors."""
    log_files = ["logs/api.log", "logs/monitoring.log", "logs/preprocessing.log"]

    results = {}
    for log_file in log_files:
        try:
            with open(log_file) as f:
                last_lines = f.readlines()[-100:]  # Last 100 lines
                errors = [line for line in last_lines if "ERROR" in line]
                results[log_file] = errors[-1] if errors else None
        except (FileNotFoundError, IndexError):
            results[log_file] = None

    return results


def display_results(
    api_health: Dict[str, bool],
    system_health: Dict[str, float],
    model_files: Dict[str, bool],
    log_errors: Dict[str, Optional[str]],
) -> None:
    """Display health check results in tables."""
    # API Health
    api_table = Table(title="API Health")
    api_table.add_column("Endpoint")
    api_table.add_column("Status")

    for endpoint, healthy in api_health.items():
        api_table.add_row(endpoint, "[green]OK" if healthy else "[red]FAIL")

    console.print(api_table)
    console.print()

    # System Health
    system_table = Table(title="System Health")
    system_table.add_column("Metric")
    system_table.add_column("Value")
    system_table.add_column("Status")

    thresholds = {"cpu_percent": 80, "memory_percent": 85, "disk_percent": 90}

    for metric, value in system_health.items():
        threshold = thresholds.get(metric, 90)
        status = "[green]OK" if value < threshold else "[red]WARNING"
        system_table.add_row(metric, f"{value:.1f}%", status)

    console.print(system_table)
    console.print()

    # Model Files
    files_table = Table(title="Model Files")
    files_table.add_column("File")
    files_table.add_column("Status")

    for file_path, exists in model_files.items():
        files_table.add_row(file_path, "[green]EXISTS" if exists else "[red]MISSING")

    console.print(files_table)
    console.print()

    # Log Errors
    log_table = Table(title="Recent Log Errors")
    log_table.add_column("Log File")
    log_table.add_column("Last Error")

    for log_file, error in log_errors.items():
        log_table.add_row(log_file, "[red]" + error if error else "[green]No errors")

    console.print(log_table)


def save_results(results: Dict, output_file: Optional[Path] = None) -> None:
    """Save health check results to file."""
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as f:
            json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bible-AI health checks")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL for API health checks"
    )
    parser.add_argument(
        "--output", type=Path, help="Output file for health check results"
    )

    args = parser.parse_args()

    # Run health checks
    api_health = check_api_health(args.url)
    system_health = check_system_health()
    model_files = check_model_files()
    log_errors = check_logs()

    # Display results
    display_results(api_health, system_health, model_files, log_errors)

    # Save results if requested
    if args.output:
        results = {
            "api_health": api_health,
            "system_health": system_health,
            "model_files": model_files,
            "log_errors": log_errors,
        }
        save_results(results, args.output)
        console.print(f"\nResults saved to {args.output}")

    # Exit with error if any checks failed
    if not all(api_health.values()):
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
