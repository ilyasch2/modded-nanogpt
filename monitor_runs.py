#!/usr/bin/env python3
"""Monitor training runs and compare spectral learning experiments."""

import glob
import json
import os
import time
from pathlib import Path

def get_latest_runs():
    """Find the three latest run directories."""
    log_dir = Path("logs")
    if not log_dir.exists():
        return []

    # Get all run directories sorted by modification time
    runs = sorted(log_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Find the three we care about
    baseline_std = None
    spectral_0 = None
    spectral_1 = None

    for run in runs[:10]:  # Check last 10 runs
        if "baseline_std_0" in run.name:
            baseline_std = run
        elif "spectral_lr_0" in run.name:
            spectral_0 = run
        elif "spectral_lr_1" in run.name:
            spectral_1 = run

    return {
        "baseline_std_0": baseline_std,
        "spectral_lr_0": spectral_0,
        "spectral_lr_1": spectral_1,
    }

def read_metrics(run_dir):
    """Read metrics from a run directory."""
    if run_dir is None or not run_dir.exists():
        return None

    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return None

    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))

    return metrics

def print_status(runs):
    """Print current status of all runs."""
    print("\n" + "="*80)
    print(f"Training Status - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    for name, run_dir in runs.items():
        print(f"\n{name}:")
        if run_dir is None:
            print("  Not found")
            continue

        print(f"  Path: {run_dir}")

        metrics = read_metrics(run_dir)
        if metrics is None or len(metrics) == 0:
            print("  Status: Starting...")
            continue

        latest = metrics[-1]
        step = latest.get('step', 0)
        val_loss = latest.get('val_loss', None)
        train_time = latest.get('train_time', 0)

        print(f"  Step: {step}/1600")
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Train Time: {train_time:.1f}s")

        # Calculate loss improvement
        if len(metrics) > 1 and val_loss is not None:
            first_val = next((m['val_loss'] for m in metrics if 'val_loss' in m), None)
            if first_val is not None:
                improvement = first_val - val_loss
                print(f"  Improvement: {improvement:.4f} (from {first_val:.4f})")

def compare_losses(runs):
    """Compare validation losses across runs."""
    print("\n" + "="*80)
    print("Loss Comparison (latest validation loss)")
    print("="*80)

    losses = {}
    for name, run_dir in runs.items():
        metrics = read_metrics(run_dir)
        if metrics:
            for m in reversed(metrics):
                if 'val_loss' in m:
                    losses[name] = m['val_loss']
                    break

    if not losses:
        print("No validation losses available yet")
        return

    # Sort by loss (best first)
    sorted_losses = sorted(losses.items(), key=lambda x: x[1])

    print(f"\n{'Rank':<6} {'Config':<25} {'Val Loss':<12} {'Δ from best'}")
    print("-" * 60)

    best_loss = sorted_losses[0][1]
    for i, (name, loss) in enumerate(sorted_losses, 1):
        delta = loss - best_loss
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        print(f"{i:<6} {name:<25} {loss:.6f}    {delta_str}")

def main():
    """Main monitoring loop."""
    print("Starting monitoring... (Ctrl+C to stop)")

    try:
        while True:
            runs = get_latest_runs()
            print_status(runs)
            compare_losses(runs)

            # Check if all runs are complete
            all_metrics = [read_metrics(r) for r in runs.values() if r is not None]
            if all_metrics and all(m and len(m) > 0 for m in all_metrics):
                latest_steps = [m[-1].get('step', 0) for m in all_metrics]
                if all(s >= 1600 for s in latest_steps):
                    print("\n" + "="*80)
                    print("All runs complete!")
                    print("="*80)
                    break

            time.sleep(30)  # Update every 30 seconds

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()
