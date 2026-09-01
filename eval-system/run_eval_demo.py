"""
run_eval_demo.py

Demonstration script showing how the evaluation harness in evaluator.py
computes Precision, Recall, F1-Score, and ROC-AUC over a dataset of
dependency update outcomes.

Since we don't have a real trained model or real historical outcome data
yet, this generates a synthetic dataset that *mimics* what a reasonably
good model's predictions would look like: predictions for updates that
actually broke the project (label 1) cluster higher, and predictions for
safe updates (label 0) cluster lower, with some noisy overlap in between
(since no model is perfect).
"""

from __future__ import annotations

import random
from typing import List, Tuple

from evaluator import compute_binary_metrics, compute_roc_auc, print_metrics_table


def generate_mock_dataset(
    num_samples: int = 100, anomaly_ratio: float = 0.3
) -> Tuple[List[int], List[float]]:
    """
    Generate synthetic data for testing.

    num_samples: Total number of mock dependency updates to simulate.
    anomaly_ratio: Proportion of updates that actually broke the code (label 1).

    Returns:
        (y_true, y_pred) -- ground truth labels and simulated model predictions.
    """
    random.seed(42)  # For reproducible results

    y_true: List[int] = []
    y_pred: List[float] = []

    for _ in range(num_samples):
        # 1. Determine if this sample broke (ground truth)
        is_breaking = 1 if random.random() < anomaly_ratio else 0
        y_true.append(is_breaking)

        # 2. Simulate model prediction:
        # A good model should assign higher scores to breaking updates.
        if is_breaking == 1:
            # Center predictions for breaking updates around ~0.75 with noise
            pred = min(1.0, max(0.0, random.normalvariate(0.75, 0.15)))
        else:
            # Center predictions for safe updates around ~0.25 with noise
            pred = min(1.0, max(0.0, random.normalvariate(0.25, 0.15)))

        y_pred.append(round(pred, 4))

    return y_true, y_pred


def main() -> None:
    # 1. Generate 100 mock updates
    print("Generating 100 mock dependency update results...")
    y_true, y_pred = generate_mock_dataset(100)

    num_breaking = sum(y_true)
    num_safe = len(y_true) - num_breaking
    print(f"  -> {num_breaking} broke the project, {num_safe} were safe.\n")

    # 2. Calculate metrics
    binary_metrics = compute_binary_metrics(y_true, y_pred, threshold=0.5)
    roc_auc = compute_roc_auc(y_true, y_pred)

    combined_metrics = {
        "precision": binary_metrics["precision"],
        "recall": binary_metrics["recall"],
        "f1": binary_metrics["f1"],
        "roc_auc": roc_auc,
    }

    # 3. Print the metrics table
    print("Evaluation results (threshold = 0.5):")
    print_metrics_table(combined_metrics)

    # Bonus: show the underlying confusion-matrix counts for context.
    print(
        f"\n(TP={binary_metrics['tp']}, FP={binary_metrics['fp']}, "
        f"FN={binary_metrics['fn']}, TN={binary_metrics['tn']})"
    )


if __name__ == "__main__":
    main()