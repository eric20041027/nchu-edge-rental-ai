#!/usr/bin/env python3
"""Quick script to check model training progress from pipeline.log"""

import json
import re
from pathlib import Path
from datetime import datetime

def parse_training_logs(log_file="pipeline.log"):
    """Parse training metrics from log file"""
    if not Path(log_file).exists():
        print("❌ Log file not found!")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract JSON-like training logs
    metrics = []
    for line in content.split('\n'):
        if line.strip().startswith('{') and ('loss' in line or 'eval' in line):
            try:
                metric = json.loads(line)
                metrics.append(metric)
            except:
                pass

    if not metrics:
        print("[WARNING] No training metrics found yet")
        return

    print("\n[TRAINING PROGRESS SUMMARY]")
    print("=" * 60)
    print()

    # Group by epoch
    epochs = {}
    for m in metrics:
        epoch = m.get('epoch', 0)
        if epoch not in epochs:
            epochs[epoch] = []
        epochs[epoch].append(m)

    # Display summary
    print("Epoch | Train Loss  | Eval Loss  | Status")
    print("------+-------------+------------+------------------")

    best_eval_loss = float('inf')
    best_epoch = 0

    for epoch in sorted(epochs.keys()):
        epoch_metrics = epochs[epoch]

        # Get last training metric and first eval metric
        train_loss = None
        eval_loss = None

        for m in epoch_metrics:
            if 'loss' in m and 'eval_loss' not in m:
                train_loss = m['loss']
            if 'eval_loss' in m:
                eval_loss = m['eval_loss']

        status = ""
        if eval_loss:
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                status = "✓ BEST"
            elif eval_loss > best_eval_loss:
                status = "↑ worse"

        train_str = f"{float(train_loss):.4f}" if train_loss else "N/A"
        eval_str = f"{float(eval_loss):.4f}" if eval_loss else "N/A"

        print(f"{epoch:4.1f} | {train_str:>11} | {eval_str:>10} | {status}")

    print()
    print("-" * 60)
    print(f"[BEST] Epoch {best_epoch:.1f} (eval_loss = {best_eval_loss:.4f})")
    print()

    # Current status
    latest = metrics[-1] if metrics else {}
    current_epoch = latest.get('epoch', 0)
    print(f"[STATUS] Current Training:")
    print(f"   Epoch: {current_epoch:.2f}/10.0")
    print(f"   Progress: {min(100, int(current_epoch*10))}%")
    print(f"   Latest Loss: {latest.get('loss', 'N/A')}")

    if current_epoch < 10:
        eta_hours = max(0.5, (10 - current_epoch) * 0.7)  # ~0.7 hours per epoch
        print(f"   ETA: ~{eta_hours:.1f} hours")

    print()
    print("=" * 60)
    print("[TIP] Use 'tail -f pipeline.log' to watch live updates")
    print()

if __name__ == "__main__":
    parse_training_logs()
