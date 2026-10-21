# VerSync Evaluation & CLI System

This directory contains the user-facing command-line tool and performance
evaluation pipeline for **VerSync** — a dependency update risk predictor.

## Overview

VerSync helps developers understand the risk of upgrading a dependency
*before* they do it. Given a repository, a dependency name, and an old/new
version pair, it produces a risk report: a numerical score, a risk category
(`SAFE` / `RISKY` / `CRITICAL`), a list of likely-affected files/functions,
and a plain-English explanation with a recommendation.

This directory holds two related but separate things:

1. The **user-facing CLI** (`cli.py` + `analyzer.py`) that runs a risk
   analysis on a single dependency update and prints a report.
2. The **evaluation pipeline** (`evaluator.py` + `run_eval_demo.py`) that
   measures how good the underlying predictions are (Precision, Recall,
   F1-Score, ROC-AUC) once we have real predictions and ground-truth
   outcomes to check them against.

Right now, `analyzer.py` is a **mock/stub** — it returns deterministic,
hash-based fake results so the rest of the system (CLI, reports, evaluation
harness) can be built and tested end-to-end before the real graph/ML model
exists. See the [Integration Guide](#integration-guide-for-team-collaboration)
below for how that gets swapped in.

## Directory Structure

| File | Purpose |
|---|---|
| **`cli.py`** | Command-line entry point. Parses arguments, calls `analyzer.analyze_update`, and prints the report as a formatted text report or as JSON. |
| **`analyzer.py`** | Risk evaluation wrapper. Currently returns a deterministic mock report based on hashing the inputs; will later call our graph-ML prediction backend. |
| **`evaluator.py`** | Evaluation metrics harness (Precision, Recall, F1-Score, ROC-AUC) implemented in pure Python — no external dependencies (no numpy/scikit-learn) required. |
| **`run_eval_demo.py`** | Demo script that generates a synthetic set of 100 mock update outcomes and runs them through `evaluator.py` to show the metrics table in action. |

---

## Getting Started / Running the Code

Requires Python 3.7+ and no third-party packages — everything here is
pure standard library.

### 1. Run the Risk Prediction CLI

Run the CLI tool using Python 3:

```bash
python cli.py --repo-path /path/to/repo --dependency pandas --old-version 2.1.4 --new-version 2.2.0
```

This prints a formatted text report with the risk score, category,
reasoning, recommendation, and a table of potentially affected files.

**JSON output format** (for CI/CD integrations, scripting, or piping into
other tools):

```bash
python cli.py --repo-path /path/to/repo --dependency pandas --old-version 2.1.4 --new-version 2.2.0 --format json
```

**All CLI flags:**

| Flag | Required | Description |
|---|---|---|
| `--repo-path` | Yes | Path to the target repository. |
| `--dependency` | Yes | Name of the dependency being updated (e.g. `spring-web`). |
| `--old-version` | Yes | Current version string (e.g. `5.3.24`). |
| `--new-version` | Yes | Target version string (e.g. `6.0.7`). |
| `--dig-path` | No | Path to a precomputed Dependency-Impact Graph JSON file (e.g. `graph-model/dig_sample_output.json`). |
| `--format` | No | `text` (default) or `json`. |

### 2. Run the Evaluation Metrics Demo

Verify the metrics harness calculations and display layout on a synthetic
dataset of 100 mock dependency updates:

```bash
python run_eval_demo.py
```

This will print how many of the 100 mock updates "broke" vs. were "safe,"
followed by a metrics table (Precision, Recall, F1-Score, ROC-AUC) and the
underlying confusion-matrix counts (TP/FP/FN/TN).

---

## Integration Guide (For Team Collaboration)

### Person 2: Integrating the Graph & ML Model

When the graph construction and ML model are ready, the mock logic in
`analyzer.py` needs to be swapped out for the real thing:

1. Open `analyzer.py`.
2. The function to replace is `analyze_update(repo_path, dependency, old_version, new_version)`.
   Internally it currently calls four mock helpers you'll be replacing:
   - `_deterministic_score(...)` → replace with your **ML model's** risk
     score prediction (still a `float` between `0.0` and `1.0`).
   - `_mock_impacted_files(...)` → replace with your **graph propagation**
     logic that walks the dependency/call graph to find real affected
     files and functions, each with a real impact probability.
   - `_categorize_risk(...)` can likely stay as-is (it just buckets a score
     into `SAFE` / `RISKY` / `CRITICAL` using `SAFE_THRESHOLD` /
     `RISKY_THRESHOLD` at the top of the file) — but feel free to tune
     those thresholds once we have real score distributions to calibrate
     against.
   - `_mock_reasoning(...)` / `_mock_recommendation(...)` → replace with
     real explanations generated from your model/graph (e.g. which
     deprecated APIs were actually detected, which breaking changes were
     found in the changelog, etc).
3. **Keep the return schema of `analyze_update` unchanged** — a dictionary
   with `dependency`, `old_version`, `new_version`, `risk_score`,
   `risk_category`, `impacted_files`, `reasoning`, and `recommendation` —
   so `cli.py` continues to work out-of-the-box with no changes needed on
   the CLI side.

### Person 1 & 2: Testing Models During Training

You can use the evaluation harness to validate your models directly from
your own training/validation scripts — no need to go through the CLI at
all:

```python
from evaluator import compute_binary_metrics, compute_roc_auc

# During your evaluation loop:
y_true = [...]  # Actual outcomes (0 or 1) from the BUMP/mined dataset
y_pred = [...]  # Probabilities from your model

metrics = compute_binary_metrics(y_true, y_pred, threshold=0.5)
auc = compute_roc_auc(y_true, y_pred)

print(metrics)   # {"precision": ..., "recall": ..., "f1": ..., "tp": ..., "fp": ..., "fn": ..., "tn": ...}
print(auc)       # e.g. 0.94
```

If you want the same nicely formatted console table the CLI uses, combine
the two results and pass them to `print_metrics_table`:

```python
from evaluator import compute_binary_metrics, compute_roc_auc, print_metrics_table

binary_metrics = compute_binary_metrics(y_true, y_pred, threshold=0.5)
roc_auc = compute_roc_auc(y_true, y_pred)

print_metrics_table({
    "precision": binary_metrics["precision"],
    "recall": binary_metrics["recall"],
    "f1": binary_metrics["f1"],
    "roc_auc": roc_auc,
})
```

See `run_eval_demo.py` for a complete, runnable example of this pattern
using synthetic data.

### 3. Run the Baseline Model & Ablation Harness

Train and evaluate the baseline models against Person 1's verified BUMP dataset:

```bash
python train_baseline.py
```

This runs a 5-fold `StratifiedGroupKFold` cross-validation (grouped by repository)
with out-of-fold probability pooling and reports a 5-arm ablation table:
- **Majority Baseline** ("always breaking")
- **SemVer only (all 59 samples)**
- **SemVer only (graph subset)**
- **Graph only (graph subset)**
- **SemVer + Graph (graph subset)**

To generate pre-update DIGs for dataset samples:
```bash
python batch_build_digs.py [--limit <N>] [--safe-only]
```
---

## Person 2: Tree Models, Structural-Graph Features and GNN (`train_models.py`)

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r eval-system/requirements.txt   # javalang, numpy, scipy, scikit-learn, xgboost, matplotlib, shap, torch
```

`shap` and `torch` are optional: SHAP importance is skipped if `shap` is missing, and
`--no-gnn` skips the GNN arm.

### 1. (Re)build DIGs for all samples

```bash
python eval-system/batch_build_digs.py                 # build missing DIGs only
python eval-system/batch_build_digs.py --rebuild       # rebuild everything
python eval-system/batch_build_digs.py --only causeway nem --build-timeout 1800
python eval-system/batch_build_digs.py --no-head-fallback   # fail instead of analyzing HEAD
```

Each run merges a per-sample row into `dataset/dig_build_report.csv` (status, failure
stage and reason, which commit was analyzed, `.java` files, unparsable files, nodes/edges,
seconds). Each DIG also carries a `build_metadata` block with the same provenance.
`checkout_mode` is `parent_commit` (SAFE samples, from Person 1's task2 results),
`breaking~1` (BREAKING samples: parent of the BUMP breaking commit, fetched from
`pull/<n>/head`) or `HEAD_fallback` (the pre-update commit was unreachable; the DIG then
describes current code, not the pre-update state).

### 2. Train and compare all arms

```bash
python eval-system/train_models.py                        # full run (~10-15 min on a laptop CPU)
python eval-system/train_models.py --seeds 1 --perm-iters 0 --bootstrap-iters 200   # quick run
python eval-system/train_models.py --no-gnn               # without torch
```

Every arm uses exactly the same protocol as `train_baseline.py` (functions are imported from it):
5-fold `StratifiedGroupKFold` grouped by project, pooled out-of-fold probabilities,
threshold 0.5, 95% bootstrap CIs over the pooled OOF predictions. The LR rows are
`train_baseline.evaluate_model_pipeline` itself, so they match `train_baseline.py`'s output.

| Arm | Features |
|---|---|
| `LR` | logistic regression baseline (balanced class weights) |
| `RF` | Random Forest, 500 trees, `min_samples_leaf=2`, `class_weight="balanced_subsample"` |
| `XGB` | XGBoost, 300 depth-2 trees, lr 0.05, `scale_pos_weight = n_safe/n_breaking` per fold |
| `GNN impact-graph+SemVer` | 2-layer GCN over the dependency's impact subgraph + SemVer features (`gnn_model.py`) |
| `MLP SemVer (GNN control)` | the same network with the graph branch removed |

Feature sets: `SemVer` (7), `Graph` (7 baseline blast-radius features), `Struct` (21
structural features from `graph_features.py`), split into `DepStruct` (16 features
describing how the *updated dependency* is used: API fan-in, call share, PageRank share,
depth distribution, modules/files reached, test share, ...) and `RepoCtx` (5 repo-level
features: graph size, density, number of declared deps, unresolved-call share).
`RepoCtx` is a confound control, not a model to deploy (see `results/README.md`).

Hyperparameters are fixed a priori. They are not tuned, because with 8 SAFE samples any
tuning on the CV folds would inflate the numbers.

Outputs in `results/`: `model_comparison.csv/json`, `seed_stability.csv`,
`label_permutation_test.json`, `oof_predictions.csv`, `feature_importance.csv/json`,
`feature_importance.png`; the interpretation is in `results/README.md`.

> Note (macOS): torch and xgboost each bundle their own `libomp`; the GNN therefore runs
> torch single-threaded to avoid an OpenMP deadlock.
