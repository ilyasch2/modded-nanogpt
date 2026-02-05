#!/usr/bin/env python3
"""
Plot training and validation loss from log files.

Usage:
    python plot_loss.py                          # plots the most recent log file
    python plot_loss.py logs/run_id.txt          # plots a specific log file
    python plot_loss.py logs/run1.txt logs/run2.txt logs/run3.txt  # compare multiple runs
"""

import re
import sys
import glob
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log_file(log_path):
    """Parse log file and extract step numbers, validation losses, learning rates, betas, and batch sizes."""
    steps = []
    val_losses = []
    train_times = []
    lr_muls = []

    # Also track all steps with lr_mul, betas, and batch_size (including training steps)
    all_steps = []
    all_lr_muls = []
    all_beta1s = []
    all_beta2s = []
    all_batch_sizes = []

    with open(log_path, 'r') as f:
        for line in f:
            # Match validation lines with betas and batch_size: step:100/1000 val_loss:3.2845 lr_mul:1.5000 beta1:0.5000 beta2:0.9500 batch_size:131072 train_time:1234ms
            val_match = re.search(r'step:(\d+)/\d+\s+val_loss:([\d.]+)\s+lr_mul:([\d.]+)\s+beta1:([\d.]+)\s+beta2:([\d.]+)\s+batch_size:(\d+)\s+train_time:([\d.]+)ms', line)
            if val_match:
                step = int(val_match.group(1))
                val_loss = float(val_match.group(2))
                lr_mul = float(val_match.group(3))
                beta1 = float(val_match.group(4))
                beta2 = float(val_match.group(5))
                batch_size = int(val_match.group(6))
                train_time = float(val_match.group(7))

                steps.append(step)
                val_losses.append(val_loss)
                lr_muls.append(lr_mul)
                train_times.append(train_time)

            # Match all lines with lr_mul, betas, and batch_size (training and validation)
            batch_match = re.search(r'step:(\d+)/\d+\s+.*?lr_mul:([\d.]+)\s+beta1:([\d.]+)\s+beta2:([\d.]+)\s+batch_size:(\d+)', line)
            if batch_match:
                all_steps.append(int(batch_match.group(1)))
                all_lr_muls.append(float(batch_match.group(2)))
                all_beta1s.append(float(batch_match.group(3)))
                all_beta2s.append(float(batch_match.group(4)))
                all_batch_sizes.append(int(batch_match.group(5)))
            else:
                # Fallback for old logs without betas and batch_size
                beta_match = re.search(r'step:(\d+)/\d+\s+.*?lr_mul:([\d.]+)\s+beta1:([\d.]+)\s+beta2:([\d.]+)', line)
                if beta_match:
                    all_steps.append(int(beta_match.group(1)))
                    all_lr_muls.append(float(beta_match.group(2)))
                    all_beta1s.append(float(beta_match.group(3)))
                    all_beta2s.append(float(beta_match.group(4)))
                else:
                    # Fallback for even older logs
                    lr_match = re.search(r'step:(\d+)/\d+\s+lr_mul:([\d.]+)', line)
                    if lr_match and int(lr_match.group(1)) not in all_steps:
                        all_steps.append(int(lr_match.group(1)))
                        all_lr_muls.append(float(lr_match.group(2)))

    return steps, val_losses, train_times, lr_muls, all_steps, all_lr_muls, all_beta1s, all_beta2s, all_batch_sizes

def plot_losses(log_paths):
    """Create plots for validation loss, learning rate, and training time.

    Args:
        log_paths: Single log path (str) or list of log paths for comparison
    """
    # Handle single path or list of paths
    if isinstance(log_paths, str):
        log_paths = [log_paths]

    # Parse all log files
    all_data = []
    for log_path in log_paths:
        steps, val_losses, train_times, lr_muls, all_steps, all_lr_muls, all_beta1s, all_beta2s, all_batch_sizes = parse_log_file(log_path)
        if not steps:
            print(f"Warning: No validation loss data found in {log_path}")
            continue
        all_data.append({
            'path': log_path,
            'name': Path(log_path).stem,  # Use filename without extension as label
            'steps': steps,
            'val_losses': val_losses,
            'train_times': [t / 1000 for t in train_times],  # Convert to seconds
            'lr_muls': lr_muls,
            'all_steps': all_steps,
            'all_lr_muls': all_lr_muls,
            'all_beta1s': all_beta1s,
            'all_beta2s': all_beta2s,
            'all_batch_sizes': all_batch_sizes,
        })

    if not all_data:
        print("No data to plot")
        return

    # Create figure with five subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 22))

    # Define colors for multiple runs
    colors = plt.cm.tab10(range(len(all_data)))

    # Plot validation loss for all runs
    for i, data in enumerate(all_data):
        marker = 'o' if len(all_data) <= 3 else ''
        markersize = 4 if len(all_data) <= 3 else 0
        ax1.plot(data['steps'], data['val_losses'],
                color=colors[i], linewidth=2, marker=marker, markersize=markersize,
                label=data['name'])

    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_ylim(3.2, 5.25)

    if len(all_data) == 1:
        ax1.set_title(f'Validation Loss Over Time\n{all_data[0]["name"]}', fontsize=14)
    else:
        ax1.set_title('Validation Loss Comparison', fontsize=14)

    ax1.grid(True, alpha=0.3)

    # Add horizontal line at 3.28 (target loss)
    ax1.axhline(y=3.28, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Target: 3.28')
    ax1.legend(loc='best', fontsize=9)

    # Add annotation for when 3.28 was reached (only for single run)
    if len(all_data) == 1:
        data = all_data[0]
        reached_328_step = None
        reached_328_time = None
        for i, loss in enumerate(data['val_losses']):
            if loss <= 3.28:
                reached_328_step = data['steps'][i]
                reached_328_time = data['train_times'][i]
                break

        final_loss = data['val_losses'][-1]
        info_text = f'Final Loss: {final_loss:.4f}'
        if reached_328_step is not None:
            info_text += f'\nReached 3.28 at step {reached_328_step}'
            info_text += f'\n({reached_328_time:.1f}s)'
        ax1.text(0.98, 0.98, info_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=11)

    # Plot training time for all runs
    for i, data in enumerate(all_data):
        marker = 's' if len(all_data) <= 3 else ''
        markersize = 4 if len(all_data) <= 3 else 0
        ax2.plot(data['steps'], data['train_times'],
                color=colors[i], linewidth=2, marker=marker, markersize=markersize,
                label=data['name'])

    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)

    if len(all_data) == 1:
        ax2.set_title('Cumulative Training Time', fontsize=14)
    else:
        ax2.set_title('Training Time Comparison', fontsize=14)

    ax2.grid(True, alpha=0.3)
    if len(all_data) > 1:
        ax2.legend(loc='best', fontsize=9)

    # Add annotations (only for single run)
    if len(all_data) == 1:
        data = all_data[0]
        # Find when 3.28 was reached
        reached_328_time = None
        for i, loss in enumerate(data['val_losses']):
            if loss <= 3.28:
                reached_328_time = data['train_times'][i]
                break

        if reached_328_time is not None:
            ax2.axhline(y=reached_328_time, color='g', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Time to 3.28: {reached_328_time:.1f}s')
            ax2.legend(loc='upper left', fontsize=10)

        total_time_sec = data['train_times'][-1]
        total_time_min = total_time_sec / 60
        ax2.text(0.98, 0.98, f'Total Time: {total_time_sec:.1f}s ({total_time_min:.2f} min)',
                transform=ax2.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=11)

    # Plot learning rate schedule for all runs
    for i, data in enumerate(all_data):
        if data['all_lr_muls']:
            ax3.plot(data['all_steps'], data['all_lr_muls'],
                    color=colors[i], linewidth=2, label=data['name'])

    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Learning Rate Multiplier (lr_mul)', fontsize=12)

    if len(all_data) == 1:
        ax3.set_title('Learning Rate Schedule', fontsize=14)
    else:
        ax3.set_title('Learning Rate Schedule Comparison', fontsize=14)

    ax3.grid(True, alpha=0.3)
    if len(all_data) > 1:
        ax3.legend(loc='best', fontsize=9)

    # Add annotations for key LR values (only for single run)
    if len(all_data) == 1 and all_data[0]['all_lr_muls']:
        lr_muls = all_data[0]['all_lr_muls']
        max_lr = max(lr_muls)
        min_lr = min(lr_muls)
        final_lr = lr_muls[-1]

        info_text = f'Max: {max_lr:.4f}\nMin: {min_lr:.4f}\nFinal: {final_lr:.4f}'
        ax3.text(0.02, 0.98, info_text,
                transform=ax3.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5),
                fontsize=11)

    # Plot beta schedules for all runs
    for i, data in enumerate(all_data):
        if data['all_beta1s'] and data['all_beta2s']:
            ax4.plot(data['all_steps'], data['all_beta1s'],
                    color=colors[i], linewidth=2, linestyle='-', label=f"{data['name']} (beta1)")
            ax4.plot(data['all_steps'], data['all_beta2s'],
                    color=colors[i], linewidth=2, linestyle='--', label=f"{data['name']} (beta2)")

    ax4.set_xlabel('Step', fontsize=12)
    ax4.set_ylabel('Beta Values', fontsize=12)

    if len(all_data) == 1:
        ax4.set_title('Adam Beta Schedule', fontsize=14)
    else:
        ax4.set_title('Adam Beta Schedule Comparison', fontsize=14)

    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=9)

    # Add annotations for key beta values (only for single run)
    if len(all_data) == 1 and all_data[0]['all_beta1s'] and all_data[0]['all_beta2s']:
        beta1s = all_data[0]['all_beta1s']
        beta2s = all_data[0]['all_beta2s']

        if beta1s and beta2s:
            final_beta1 = beta1s[-1]
            final_beta2 = beta2s[-1]

            info_text = f'Final beta1: {final_beta1:.4f}\nFinal beta2: {final_beta2:.4f}'
            ax4.text(0.02, 0.98, info_text,
                    transform=ax4.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
                    fontsize=11)

    # Plot batch size schedules for all runs
    for i, data in enumerate(all_data):
        if data['all_batch_sizes']:
            ax5.plot(data['all_steps'], data['all_batch_sizes'],
                    color=colors[i], linewidth=2, drawstyle='steps-post', label=data['name'])

    ax5.set_xlabel('Step', fontsize=12)
    ax5.set_ylabel('Batch Size', fontsize=12)

    if len(all_data) == 1:
        ax5.set_title('Batch Size Schedule', fontsize=14)
    else:
        ax5.set_title('Batch Size Schedule Comparison', fontsize=14)

    ax5.grid(True, alpha=0.3)
    if len(all_data) > 1:
        ax5.legend(loc='best', fontsize=9)

    # Add annotations for batch size transitions (only for single run)
    if len(all_data) == 1 and all_data[0]['all_batch_sizes']:
        batch_sizes = all_data[0]['all_batch_sizes']

        if batch_sizes:
            unique_batch_sizes = sorted(set(batch_sizes))
            info_text = 'Batch sizes:\n' + '\n'.join([f'{bs:,}' for bs in unique_batch_sizes])
            ax5.text(0.02, 0.98, info_text,
                    transform=ax5.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
                    fontsize=11)

    plt.tight_layout()

    # Save the plot
    if len(all_data) == 1:
        output_path = Path(all_data[0]['path']).with_suffix('.png')
    else:
        output_path = Path('logs/comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also display if in interactive mode
    plt.show()

    # Print summary statistics for each run
    for data in all_data:
        print(f"\n=== Training Summary: {data['name']} ===")
        print(f"Total steps: {data['steps'][-1] if data['steps'] else 0}")
        print(f"Initial val loss: {data['val_losses'][0]:.4f}" if data['val_losses'] else "N/A")
        print(f"Final val loss: {data['val_losses'][-1]:.4f}" if data['val_losses'] else "N/A")
        if len(data['val_losses']) > 1:
            improvement = data['val_losses'][0] - data['val_losses'][-1]
            print(f"Loss improvement: {improvement:.4f}")

        # Find when 3.28 was reached
        reached_328_step = None
        reached_328_time = None
        for i, loss in enumerate(data['val_losses']):
            if loss <= 3.28:
                reached_328_step = data['steps'][i]
                reached_328_time = data['train_times'][i]
                break

        if reached_328_step is not None:
            print(f"Reached 3.28 loss at step: {reached_328_step}")
            print(f"Time to reach 3.28: {reached_328_time:.1f}s ({reached_328_time/60:.2f} min)")
        else:
            print("Target loss of 3.28 not yet reached")

        if data['train_times']:
            total_time = data['train_times'][-1]
            print(f"Total training time: {total_time:.1f}s ({total_time/60:.2f} min)")

def get_most_recent_log():
    """Find the most recent log file in the logs directory."""
    log_files = glob.glob("logs/*.txt")
    if not log_files:
        print("No log files found in logs/ directory")
        sys.exit(1)

    # Sort by modification time, most recent first
    log_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return log_files[0]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Multiple log files provided
        log_paths = sys.argv[1:]
        # Check all files exist
        for log_path in log_paths:
            if not Path(log_path).exists():
                print(f"Error: Log file not found: {log_path}")
                sys.exit(1)
        if len(log_paths) == 1:
            print(f"Plotting: {log_paths[0]}")
        else:
            print(f"Comparing {len(log_paths)} runs:")
            for log_path in log_paths:
                print(f"  - {log_path}")
    else:
        # No arguments - use most recent log
        log_paths = [get_most_recent_log()]
        print(f"Using most recent log file: {log_paths[0]}")

    plot_losses(log_paths)
