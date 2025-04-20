#!/usr/bin/env python3
"""Benchmark script for Bible-AI performance monitoring."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from rich.console import Console
from rich.table import Table

console = Console()

def run_api_benchmark(url: str, iterations: int = 100) -> Dict[str, float]:
    """Run API endpoint benchmarks."""
    endpoints = {
        "health": "/health",
        "verse_lookup": "/api/v1/verse/John 3:16",
        "search": "/api/v1/search?q=love",
        "theological": "/api/v1/theological?q=salvation",
    }
    
    results = {}
    for name, path in endpoints.items():
        times: List[float] = []
        for _ in range(iterations):
            start = time.time()
            try:
                response = requests.get(f"{url}{path}", timeout=30)
                response.raise_for_status()
                times.append(time.time() - start)
            except requests.RequestException as e:
                console.print(f"[red]Error benchmarking {name}: {e}")
                break
        
        if times:
            results[name] = {
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "p95": sorted(times)[int(len(times) * 0.95)],
            }
    
    return results

def run_model_benchmark(iterations: int = 10) -> Dict[str, float]:
    """Run model inference benchmarks."""
    try:
        from src.model.verse_embeddings import VerseEmbedder
        from src.theology.validator import TheologicalValidator
        
        results = {}
        
        # Test embedding generation
        embedder = VerseEmbedder()
        texts = ["For God so loved the world", "The Lord is my shepherd"]
        start = time.time()
        for _ in range(iterations):
            embedder.model.encode(texts)
        embed_time = (time.time() - start) / iterations
        results["embedding"] = embed_time
        
        # Test theological validation
        validator = TheologicalValidator()
        statements = [
            "God exists in three persons",
            "Jesus Christ is fully God and fully man",
            "The Bible is inspired by God"
        ]
        start = time.time()
        for _ in range(iterations):
            validator.validate_batch(statements)
        validate_time = (time.time() - start) / iterations
        results["theological"] = validate_time
        
        return results
    except ImportError as e:
        console.print(f"[yellow]Skipping model benchmarks: {e}")
        return {}

def save_results(results: Dict[str, Dict[str, float]], output_file: Path) -> None:
    """Save benchmark results to file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump({
            "timestamp": time.time(),
            "results": results
        }, f, indent=2)

def display_results(results: Dict[str, Dict[str, float]]) -> None:
    """Display benchmark results in a table."""
    table = Table(title="Bible-AI Benchmark Results")
    table.add_column("Component")
    table.add_column("Average (s)")
    table.add_column("Min (s)")
    table.add_column("Max (s)")
    table.add_column("P95 (s)")
    
    for component, metrics in results.items():
        if isinstance(metrics, dict) and "avg" in metrics:
            table.add_row(
                component,
                f"{metrics['avg']:.3f}",
                f"{metrics['min']:.3f}",
                f"{metrics['max']:.3f}",
                f"{metrics['p95']:.3f}",
            )
        else:
            table.add_row(component, f"{metrics:.3f}", "-", "-", "-")
    
    console.print(table)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bible-AI benchmarks")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Base URL for API benchmarks")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations for each benchmark")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmarks with fewer iterations")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/output.json"),
                       help="Output file for benchmark results")
    
    args = parser.parse_args()
    
    if args.quick:
        args.iterations = 10
    
    console.print(f"Running benchmarks with {args.iterations} iterations...")
    
    # Run benchmarks
    api_results = run_api_benchmark(args.url, args.iterations)
    model_results = run_model_benchmark(args.iterations)
    
    # Combine results
    all_results = {**api_results, **model_results}
    
    # Display results
    display_results(all_results)
    
    # Save results
    save_results(all_results, args.output)
    console.print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()