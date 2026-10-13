"""
train_baseline.py

VerSync Baseline Risk Model & Cross-Validation Evaluation Harness.

Connects Person 1's verified dataset to the evaluation pipeline:
  1. Featurizes update events using static SemVer features and (when available)
     DIG graph traversal features from analyzer.py.
  2. Evaluates via 5-fold StratifiedGroupKFold cross-validation (grouped by project)
     to prevent multi-sample repos (e.g. html2pop3, micro-migration) from leaking
     across train and test splits.
  3. Pools out-of-fold predicted probabilities across all folds before metric calculation
     to ensure stable evaluation with scarce negative samples (8 safe vs 51 breaking).
  4. Reports a multi-arm ablation table:
       - Majority Baseline ("always breaking")
       - SemVer only (all 59 samples)
       - SemVer only (graph subset, if available)
       - Graph only (graph subset, if available)
       - SemVer + Graph (graph subset, if available)
  5. Computes Precision, Recall, F1, ROC-AUC, Balanced Accuracy, MCC, Safe-class Recall,
     and 95% Bootstrap Confidence Intervals.

Usage:
    python eval-system/train_baseline.py [--dataset <path>] [--dig-dir <path>] [--bootstrap-iters <int>]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Re-use existing evaluation helpers from evaluator.py & analyzer.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import compute_binary_metrics, compute_roc_auc
from analyzer import classify_semver_bump, extract_graph_features, parse_semver


# ---------------------------------------------------------------------------
# Metric helpers (Balanced Accuracy, MCC, Safe Recall)
# ---------------------------------------------------------------------------

def compute_extended_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics, ROC-AUC, balanced accuracy, MCC,
    and safe-class recall from true labels and predicted probabilities.
    """
    y_pred = (y_prob >= threshold).astype(int)
    bin_m = compute_binary_metrics(y_true.tolist(), y_pred.tolist(), threshold=threshold)
    auc = compute_roc_auc(y_true.tolist(), y_prob.tolist())

    tp, fp, fn, tn = bin_m["tp"], bin_m["fp"], bin_m["fn"], bin_m["tn"]

    # Balanced accuracy = (Sensitivity + Specificity) / 2
    rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (rec_pos + rec_neg) / 2.0

    # Matthews Correlation Coefficient (MCC)
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    # Safe class metrics (class 0)
    safe_recall = rec_neg
    safe_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    safe_f1 = (
        (2.0 * safe_prec * safe_recall) / (safe_prec + safe_recall)
        if (safe_prec + safe_recall) > 0
        else 0.0
    )

    return {
        "precision": bin_m["precision"],
        "recall": bin_m["recall"],
        "f1": bin_m["f1"],
        "roc_auc": round(auc, 4),
        "balanced_acc": round(balanced_acc, 4),
        "mcc": round(mcc, 4),
        "safe_recall": round(safe_recall, 4),
        "safe_precision": round(safe_prec, 4),
        "safe_f1": round(safe_f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_key: str,
    n_iters: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval [lower, upper] for a metric."""
    if n_iters <= 0 or len(y_true) < 5:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    n = len(y_true)
    values = []

    for _ in range(n_iters):
        indices = rng.randint(0, n, size=n)
        sample_y = y_true[indices]
        sample_prob = y_prob[indices]
        # Skip if sample lacks both positive and negative classes
        if len(np.unique(sample_y)) < 2:
            continue
        try:
            m = compute_extended_metrics(sample_y, sample_prob, threshold=threshold)
            values.append(m[metric_key])
        except Exception:
            continue

    if not values:
        return 0.0, 0.0

    lower = float(np.percentile(values, 2.5))
    upper = float(np.percentile(values, 97.5))
    return round(lower, 3), round(upper, 3)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_semver_features_row(old_v: str, new_v: str) -> Dict[str, float]:
    """Extract strictly outcome-independent SemVer features."""
    old_maj, old_min, old_pat = parse_semver(old_v)
    new_maj, new_min, new_pat = parse_semver(new_v)
    bump_type = classify_semver_bump(old_v, new_v)

    maj_delta = max(0, new_maj - old_maj)
    min_delta = abs(new_min - old_min)
    pat_delta = abs(new_pat - old_pat)
    distance = (maj_delta * 4.0) + (min_delta * 2.0) + (pat_delta * 1.0)

    return {
        "is_major": 1.0 if bump_type == "MAJOR" else 0.0,
        "is_minor": 1.0 if bump_type == "MINOR" else 0.0,
        "is_patch": 1.0 if bump_type == "PATCH" else 0.0,
        "maj_delta": float(maj_delta),
        "min_delta": float(min_delta),
        "pat_delta": float(pat_delta),
        "version_distance": float(distance),
    }


def find_dig_for_sample(
    project: str,
    source_json: str,
    dig_dir: Optional[str],
) -> Optional[dict]:
    """Look up a pre-generated DIG JSON file for this sample in dig_dir."""
    if not dig_dir or not os.path.isdir(dig_dir):
        return None

    stem = os.path.splitext(source_json)[0] if source_json else ""
    candidates = [
        os.path.join(dig_dir, f"{project}_{stem}.json"),
        os.path.join(dig_dir, f"{project}.json"),
        os.path.join(dig_dir, f"{stem}.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            try:
                with open(c, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                continue
    return None


def extract_graph_features_row(
    dig_data: Optional[dict],
    dep: str,
    old_v: str,
    new_v: str,
) -> Dict[str, float]:
    """
    Extract log-transformed blast radius and normalized call-graph features
    from a DIG. Zero-usage is tracked cleanly as dep_found_in_graph.
    """
    if dig_data is None:
        return {
            "has_graph": 0.0,
            "dep_found_in_graph": 0.0,
            "direct_calls_log": 0.0,
            "indirect_calls_log": 0.0,
            "affected_functions_log": 0.0,
            "affected_files_log": 0.0,
            "max_depth": 0.0,
            "ratio_affected_files": 0.0,
        }

    gf = extract_graph_features(dig_data, dep, old_v, new_v)
    dep_found = 1.0 if gf["matched_dep_nodes"] else 0.0

    total_files = sum(1 for n in dig_data.get("nodes", []) if n.get("type") == "FILE")
    ratio_files = (gf["affected_files"] / total_files) if total_files > 0 else 0.0

    return {
        "has_graph": 1.0,
        "dep_found_in_graph": dep_found,
        "direct_calls_log": float(np.log1p(gf["direct_calls"])),
        "indirect_calls_log": float(np.log1p(gf["indirect_calls"])),
        "affected_functions_log": float(np.log1p(gf["affected_functions"])),
        "affected_files_log": float(np.log1p(gf["affected_files"])),
        "max_depth": float(gf["max_depth"]),
        "ratio_affected_files": float(min(1.0, ratio_files)),
    }


# ---------------------------------------------------------------------------
# Dataset Loading & Featurization
# ---------------------------------------------------------------------------

def load_and_featurize_dataset(
    csv_path: str,
    dig_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load final_verified_dataset.csv, extract features for each row, and return:
      rows, y (labels: 1=BREAKING, 0=SAFE), groups (project names), and feature dicts.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    y_list = []
    groups_list = []
    feature_dicts = []

    for r in rows:
        label = 1 if r["classification"].strip().upper() == "BREAKING" else 0
        y_list.append(label)
        groups_list.append(r["project"].strip())

        dep = f"{r['dependency_group_id']}:{r['dependency_artifact_id']}"
        old_v = r["old_version"]
        new_v = r["new_version"]
        src_json = r.get("source_json", "")

        semver_f = extract_semver_features_row(old_v, new_v)
        dig_data = find_dig_for_sample(r["project"], src_json, dig_dir)
        graph_f = extract_graph_features_row(dig_data, dep, old_v, new_v)

        combined = {**semver_f, **graph_f}
        feature_dicts.append(combined)

    return rows, np.array(y_list), np.array(groups_list), feature_dicts


# ---------------------------------------------------------------------------
# Cross-Validation Evaluator
# ---------------------------------------------------------------------------

def evaluate_model_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run StratifiedGroupKFold cross-validation.
    Returns out-of-fold predicted probabilities and pooled metrics.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in sgkf.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]

        # Pipeline: StandardScaler inside fold, LogisticRegression with balanced weights
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)),
        ])
        pipe.fit(X_train, y_train)

        # Get probabilities for class 1 (BREAKING)
        probs = pipe.predict_proba(X_val)
        # Handle cases where training fold only has 1 class (defensive)
        if probs.shape[1] == 2:
            oof_probs[val_idx] = probs[:, 1]
        else:
            oof_probs[val_idx] = 1.0 if pipe.classes_[0] == 1 else 0.0

    metrics = compute_extended_metrics(y, oof_probs)
    return oof_probs, metrics


def evaluate_majority_baseline(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Trivial baseline: always predicts the majority class ("BREAKING" / 1)
    with probability equal to the class proportion.
    """
    majority_prob = float(np.mean(y == 1))
    oof_probs = np.full(len(y), majority_prob, dtype=float)
    metrics = compute_extended_metrics(y, oof_probs, threshold=0.5)
    return oof_probs, metrics


# ---------------------------------------------------------------------------
# Table Formatting & Main Pipeline
# ---------------------------------------------------------------------------

def format_row(name: str, m: Dict[str, float], cis: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
    """Format an arm row with optional 95% bootstrap confidence interval."""
    def cell(val: float, key: str) -> str:
        s = f"{val:.3f}"
        if cis and key in cis:
            low, high = cis[key]
            s += f" [{low:.2f},{high:.2f}]"
        return s

    p_str = cell(m["precision"], "precision")
    r_str = cell(m["recall"], "recall")
    f_str = cell(m["f1"], "f1")
    auc_str = cell(m["roc_auc"], "roc_auc")
    bacc_str = cell(m["balanced_acc"], "balanced_acc")
    mcc_str = cell(m["mcc"], "mcc")
    srec_str = cell(m["safe_recall"], "safe_recall")

    return (
        f"| {name.ljust(30)} | {p_str.ljust(15)} | {r_str.ljust(15)} | {f_str.ljust(15)} "
        f"| {auc_str.ljust(15)} | {bacc_str.ljust(15)} | {mcc_str.ljust(15)} | {srec_str.ljust(17)} |"
    )


def run_experiment(
    dataset_path: str,
    dig_dir: Optional[str] = None,
    bootstrap_iters: int = 1000,
) -> None:
    print(f"\n[VerSync] Loading dataset from: {dataset_path}")
    rows, y, groups, feat_dicts = load_and_featurize_dataset(dataset_path, dig_dir)

    n_total = len(y)
    n_breaking = int(np.sum(y == 1))
    n_safe = int(np.sum(y == 0))
    n_groups = len(set(groups))
    n_graphs = sum(1 for fd in feat_dicts if fd.get("has_graph", 0.0) == 1.0)

    print(f"[VerSync] Total samples: {n_total} ({n_breaking} BREAKING, {n_safe} SAFE)")
    print(f"[VerSync] Distinct repository groups: {n_groups}")
    print(f"[VerSync] Samples with real DIG graph: {n_graphs} / {n_total}")
    print("-" * 140)

    # 1. Feature column names
    semver_cols = ["is_major", "is_minor", "is_patch", "maj_delta", "min_delta", "pat_delta", "version_distance"]
    graph_cols = [
        "dep_found_in_graph",
        "direct_calls_log",
        "indirect_calls_log",
        "affected_functions_log",
        "affected_files_log",
        "max_depth",
        "ratio_affected_files",
    ]

    # Convert to numpy arrays
    X_semver_all = np.array([[fd[c] for c in semver_cols] for fd in feat_dicts])

    # 2. Arm 1: Majority Baseline
    maj_probs, maj_metrics = evaluate_majority_baseline(y)
    maj_cis = {
        k: compute_bootstrap_ci(y, maj_probs, k, n_iters=bootstrap_iters)
        for k in ["precision", "recall", "f1", "roc_auc", "balanced_acc", "mcc", "safe_recall"]
    }

    # 3. Arm 2: SemVer only (all samples)
    semver_probs, semver_metrics = evaluate_model_pipeline(X_semver_all, y, groups)
    semver_cis = {
        k: compute_bootstrap_ci(y, semver_probs, k, n_iters=bootstrap_iters)
        for k in ["precision", "recall", "f1", "roc_auc", "balanced_acc", "mcc", "safe_recall"]
    }

    # Print Table Header
    header = (
        f"| {'Model Arm'.ljust(30)} | {'Precision'.ljust(15)} | {'Recall'.ljust(15)} | {'F1'.ljust(15)} "
        f"| {'ROC-AUC'.ljust(15)} | {'Balanced Acc'.ljust(15)} | {'MCC'.ljust(15)} | {'Safe-class Recall'.ljust(17)} |"
    )
    sep = f"|{'-' * 32}|{'-' * 17}|{'-' * 17}|{'-' * 17}|{'-' * 17}|{'-' * 17}|{'-' * 17}|{'-' * 19}|"

    print("\n" + "=" * 140)
    print("                           VERSYNC BASELINE RISK MODEL EVALUATION REPORT")
    print("=" * 140)
    print(header)
    print(sep)
    print(format_row("Majority (always breaking)", maj_metrics, maj_cis))
    print(format_row(f"SemVer only (all {n_total})", semver_metrics, semver_cis))

    # 4. Graph Subset Arms (3, 4, 5) if any real graphs exist
    if n_graphs >= 4:
        sub_idx = [i for i, fd in enumerate(feat_dicts) if fd.get("has_graph", 0.0) == 1.0]
        y_sub = y[sub_idx]
        groups_sub = groups[sub_idx]
        X_semver_sub = X_semver_all[sub_idx]
        X_graph_sub = np.array([[feat_dicts[i][c] for c in graph_cols] for i in sub_idx])
        X_both_sub = np.hstack([X_semver_sub, X_graph_sub])

        # Arm 3: SemVer only on graph subset
        s_sub_probs, s_sub_m = evaluate_model_pipeline(X_semver_sub, y_sub, groups_sub)
        s_sub_cis = {k: compute_bootstrap_ci(y_sub, s_sub_probs, k, n_iters=bootstrap_iters) for k in s_sub_m}
        print(format_row(f"SemVer only (subset n={n_graphs})", s_sub_m, s_sub_cis))

        # Arm 4: Graph only on graph subset
        g_sub_probs, g_sub_m = evaluate_model_pipeline(X_graph_sub, y_sub, groups_sub)
        g_sub_cis = {k: compute_bootstrap_ci(y_sub, g_sub_probs, k, n_iters=bootstrap_iters) for k in g_sub_m}
        print(format_row(f"Graph only (subset n={n_graphs})", g_sub_m, g_sub_cis))

        # Arm 5: SemVer + Graph on graph subset
        both_probs, both_m = evaluate_model_pipeline(X_both_sub, y_sub, groups_sub)
        both_cis = {k: compute_bootstrap_ci(y_sub, both_probs, k, n_iters=bootstrap_iters) for k in both_m}
        print(format_row(f"SemVer + Graph (n={n_graphs})", both_m, both_cis))
    else:
        print(
            f"| {'Graph Arms (subset)'.ljust(30)} | "
            f"{'Pending batch DIG generation (run batch_build_digs.py first)'.ljust(111)} |"
        )

    print("=" * 140)
    print("\nNotes:")
    print("  * Evaluated via 5-fold StratifiedGroupKFold grouped by project; probabilities pooled out-of-fold.")
    print("  * 95% Bootstrap Confidence Intervals [lower, upper] computed across 1,000 resamples.")
    print("  * Framing: Proof-of-pipeline baseline on initial N=59 dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate VerSync baseline risk models.")
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "..", "person1", "final_verified_dataset.csv"),
        help="Path to final_verified_dataset.csv",
    )
    parser.add_argument(
        "--dig-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "dataset", "digs"),
        help="Directory containing pre-generated DIG JSON files (e.g. dataset/digs/)",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for 95% confidence intervals (default: 1000).",
    )
    args = parser.parse_args()

    run_experiment(
        dataset_path=os.path.abspath(args.dataset),
        dig_dir=os.path.abspath(args.dig_dir) if os.path.exists(args.dig_dir) else None,
        bootstrap_iters=args.bootstrap_iters,
    )


if __name__ == "__main__":
    main()
