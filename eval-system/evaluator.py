"""
evaluator.py

Pure-Python evaluation metrics harness for the dependency update risk
predictor. Computes Precision, Recall, F1-Score, and ROC-AUC by comparing
model predictions against ground truth labels (did the update actually
break the project: 1, or not: 0).

Deliberately has zero third-party dependencies (no numpy / scikit-learn)
so it works regardless of what the user has installed.
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true: List[int], y_pred: List[float], threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1-Score using a threshold to binarize predictions.

    y_true: List of ground truths (0 or 1)
    y_pred: List of prediction probabilities (0.0 to 1.0)
    threshold: Threshold to classify a prediction as breaking (1)

    Returns a dict with keys: "precision", "recall", "f1", plus the raw
    confusion-matrix counts "tp", "fp", "fn", "tn" for transparency/debugging.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length (got {len(y_true)} and {len(y_pred)})"
        )
    if len(y_true) == 0:
        raise ValueError("y_true / y_pred must contain at least one sample")

    # 1. Binarize predictions based on the threshold
    y_binarized = [1 if p >= threshold else 0 for p in y_pred]

    # 2. Count True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
    tp = fp = fn = tn = 0
    for truth, pred in zip(y_true, y_binarized):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1
        else:  # truth == 0 and pred == 0
            tn += 1

    # 3. Calculate Precision, Recall, F1-Score (guarding against division by zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------------------------------------------------------
# ROC-AUC via Mann-Whitney U rank-sum method
# ---------------------------------------------------------------------------

def _rank_predictions(y_pred: List[float]) -> List[float]:
    """
    Assign ranks (1-indexed, ascending) to predictions, averaging ranks for
    tied values as required by the Mann-Whitney U formulation.

    Example: predictions [0.1, 0.4, 0.4, 0.9] -> ranks [1, 2.5, 2.5, 4]
    """
    n = len(y_pred)
    # Sort indices by prediction value, ascending.
    sorted_indices = sorted(range(n), key=lambda i: y_pred[i])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find the extent of the run of tied values.
        while j + 1 < n and y_pred[sorted_indices[j + 1]] == y_pred[sorted_indices[i]]:
            j += 1
        # Positions i..j (0-indexed) correspond to ranks (i+1)..(j+1) (1-indexed).
        # Tied elements all receive the average of those ranks.
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        i = j + 1

    return ranks


def compute_roc_auc(y_true: List[int], y_pred: List[float]) -> float:
    """
    Calculate ROC-AUC using the Mann-Whitney U rank-sum method:

        AUC = (R_pos - P(P + 1) / 2) / (P * N)

    where P = number of positive samples, N = number of negative samples,
    and R_pos = sum of ranks of positive samples (ranks assigned over all
    predictions sorted ascending, with ties averaged).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length (got {len(y_true)} and {len(y_pred)})"
        )
    if len(y_true) == 0:
        raise ValueError("y_true / y_pred must contain at least one sample")

    p = sum(1 for label in y_true if label == 1)
    n = len(y_true) - p

    # 1. Handle edge cases: if only one class is present, AUC is undefined.
    # Convention: return 0.0 since there's nothing to discriminate between.
    if p == 0 or n == 0:
        return 0.0

    # 2. Sort predictions ascending and assign ranks (averaging ties)
    ranks = _rank_predictions(y_pred)

    # 3. Sum the ranks of positive class elements
    r_pos = sum(rank for rank, label in zip(ranks, y_true) if label == 1)

    # 4. Compute AUC using the Rank-Sum formula
    auc = (r_pos - (p * (p + 1)) / 2.0) / (p * n)

    return round(auc, 4)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: Dict[str, float]) -> None:
    """
    Print a clean ASCII table of the computed metrics (Precision, Recall, F1, ROC-AUC).

    Expects a dict that may contain any of: "precision", "recall", "f1", "roc_auc"
    (missing keys are simply skipped, so this works whether you pass metrics
    from compute_binary_metrics, compute_roc_auc, or a merged dict of both).
    """
    display_order = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
        ("roc_auc", "ROC-AUC"),
    ]

    rows = [(label, metrics[key]) for key, label in display_order if key in metrics]

    if not rows:
        print("No metrics to display.")
        return

    metric_width = max(len("Metric"), max(len(label) for label, _ in rows))
    value_width = max(len("Value"), max(len(f"{value:.4f}") for _, value in rows))

    border = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
    header = f"| {'Metric'.ljust(metric_width)} | {'Value'.ljust(value_width)} |"

    print(border)
    print(header)
    print(border)
    for label, value in rows:
        print(f"| {label.ljust(metric_width)} | {f'{value:.4f}'.ljust(value_width)} |")
    print(border)


if __name__ == "__main__":
    # Quick manual smoke test / usage example.
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [0.92, 0.15, 0.75, 0.40, 0.55, 0.10, 0.60, 0.35]

    binary_metrics = compute_binary_metrics(y_true, y_pred, threshold=0.5)
    auc = compute_roc_auc(y_true, y_pred)

    combined = {
        "precision": binary_metrics["precision"],
        "recall": binary_metrics["recall"],
        "f1": binary_metrics["f1"],
        "roc_auc": auc,
    }

    print_metrics_table(combined)