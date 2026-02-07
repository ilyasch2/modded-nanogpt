#!/usr/bin/env python3
"""
Plot spectral learning sweep results with spectral ratio evolution.

Usage:
    python plot_spectral_sweep.py  # plots all spectral_lr_* runs
"""

import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_spectral_log(log_path):
    """Parse log file and extract spectral ratios (rho values)."""
    steps = []
    val_losses = []
    spec_attn_mean = []
    spec_mlp_mean = []

    with open(log_path, 'r') as f:
        for line in f:
            if 'val_loss:' in line:
                step_match = re.search(r'step:(\d+)/', line)
                loss_match = re.search(r'val_loss:([\d.]+)', line)

                if step_match and loss_match:
                    step = int(step_match.group(1))
                    loss = float(loss_match.group(1))
                    steps.append(step)
                    val_losses.append(loss)

                    # Extract spectral ratios
                    attn_match = re.search(r'spec_attn:\[([0-9eE+.,\-\s]+)\]', line)
                    mlp_match = re.search(r'spec_mlp:\[([0-9eE+.,\-\s]+)\]', line)

                    if attn_match:
                        attn_vals = [float(x) for x in attn_match.group(1).split(',')]
                        spec_attn_mean.append(np.mean(np.abs(attn_vals)))
                    else:
                        spec_attn_mean.append(np.nan)

                    if mlp_match:
                        mlp_vals = [float(x) for x in mlp_match.group(1).split(',')]
                        spec_mlp_mean.append(np.mean(np.abs(mlp_vals)))
                    else:
                        spec_mlp_mean.append(np.nan)

    return np.array(steps), np.array(val_losses), np.array(spec_attn_mean), np.array(spec_mlp_mean)

def extract_spectral_lr(filename):
    """Extract spectral_lr value from filename."""
    match = re.search(r'spectral_lr_([\d.]+)', filename)
    if match:
        return float(match.group(1))
    return None

def main():
    # Find all spectral_lr runs (including the new sweep)
    log_files = glob.glob("logs/*spectral_lr_*-*.txt")

    if not log_files:
        print("No spectral_lr log files found")
        return

    # Parse all logs
    runs = []
    for log_file in log_files:
        spectral_lr = extract_spectral_lr(log_file)
        if spectral_lr is None:
            continue

        steps, losses, attn_mean, mlp_mean = parse_spectral_log(log_file)
        if len(steps) == 0:
            continue

        runs.append({
            'spectral_lr': spectral_lr,
            'path': log_file,
            'steps': steps,
            'losses': losses,
            'attn_mean': attn_mean,
            'mlp_mean': mlp_mean,
            'final_loss': losses[-1] if len(losses) > 0 else np.inf,
        })

    # Sort by spectral_lr
    runs.sort(key=lambda x: x['spectral_lr'])

    if len(runs) == 0:
        print("No valid runs found")
        return

    print(f"Found {len(runs)} spectral_lr runs:")
    for run in runs:
        print(f"  spectral_lr={run['spectral_lr']:.2f}: {Path(run['path']).name}")

    # Create 3-panel plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs)))

    # Panel 1: Validation Loss
    for i, run in enumerate(runs):
        label = f"spectral_lr={run['spectral_lr']:.2f} (final: {run['final_loss']:.4f})"
        ax1.plot(run['steps'], run['losses'], 'o-',
                color=colors[i], label=label, markersize=3, alpha=0.8)

    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Spectral LR Sweep: Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=3.28, color='gray', linestyle='--', alpha=0.5, label='Target: 3.28')

    # Panel 2: Mean Spectral Ratio (Attention layers)
    for i, run in enumerate(runs):
        mask = ~np.isnan(run['attn_mean'])
        if mask.sum() > 0:
            ax2.plot(run['steps'][mask], run['attn_mean'][mask], 'o-',
                    color=colors[i], label=f"spectral_lr={run['spectral_lr']:.2f}",
                    markersize=2, alpha=0.8)

    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Mean |ρ| (Attention)', fontsize=12)
    ax2.set_title('Spectral Ratio Evolution: Attention Layers', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Panel 3: Mean Spectral Ratio (MLP layers)
    for i, run in enumerate(runs):
        mask = ~np.isnan(run['mlp_mean'])
        if mask.sum() > 0:
            ax3.plot(run['steps'][mask], run['mlp_mean'][mask], 'o-',
                    color=colors[i], label=f"spectral_lr={run['spectral_lr']:.2f}",
                    markersize=2, alpha=0.8)

    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Mean |ρ| (MLP)', fontsize=12)
    ax3.set_title('Spectral Ratio Evolution: MLP Layers', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    plt.tight_layout()

    # Save plot
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_path = f'logs/spectral_sweep_{len(runs)}runs_{timestamp}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SPECTRAL LR SWEEP RESULTS")
    print("="*80)
    print(f"{'spectral_lr':^12} | {'Final Loss':^12} | {'Δ vs Best':^12} | {'Status':^15}")
    print("-"*80)

    best_loss = min(run['final_loss'] for run in runs)
    best_run = min(runs, key=lambda x: x['final_loss'])

    for run in runs:
        delta = run['final_loss'] - best_loss
        status = "🏆 BEST" if run == best_run else ("✅ Good" if delta < 0.01 else "❌ Worse")
        print(f"{run['spectral_lr']:^12.2f} | {run['final_loss']:^12.6f} | {delta:^12.6f} | {status:^15}")

    print("="*80)
    print(f"\n🏆 Best configuration: spectral_lr={best_run['spectral_lr']:.2f} (loss: {best_run['final_loss']:.6f})")

    # Check if any non-zero value beats zero
    zero_run = next((r for r in runs if r['spectral_lr'] == 0.0), None)
    if zero_run:
        better_runs = [r for r in runs if r['spectral_lr'] > 0 and r['final_loss'] < zero_run['final_loss']]
        if better_runs:
            print(f"\n✅ Found {len(better_runs)} configurations that beat spectral_lr=0:")
            for r in better_runs:
                improvement = zero_run['final_loss'] - r['final_loss']
                print(f"   spectral_lr={r['spectral_lr']:.2f}: {improvement:.6f} better")
        else:
            print(f"\n❌ No non-zero spectral_lr beats spectral_lr=0")
            print(f"   Conclusion: Spectral learning does not help for this task")

    plt.show()

if __name__ == "__main__":
    main()
