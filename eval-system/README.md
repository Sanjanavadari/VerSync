# VerSync Evaluation & CLI System

This directory contains the user-facing command-line tool and the evaluation pipeline for **VerSync**,
a dependency update risk predictor. Results produced by these scripts are written to [`../results`](../results/README.md).

## Overview

Given a repository, a dependency and an old/new version, VerSync reports how risky the update is
(`SAFE` / `RISKY` / `CRITICAL`), which functions and files are likely affected, and what to do next.
It does this by walking the repository's **Dependency-Impact Graph (DIG)** backwards from the updated
dependency to the project code that calls it, and combining the size of that blast radius with the SemVer
bump type.

This directory holds two related things:

1. The **CLI** (`cli.py` + `analyzer.py`): analyse one dependency update and print a report.
2. The **evaluation pipeline** (`train_baseline.py`, `train_models.py`, `evaluate_localization.py`,
   `calibrate_triage.py`, `evaluator.py`): measure how good the predictions are against labelled updates.

The analyzer scores updates with a hand-written heuristic (`_compute_graph_heuristic_score`), not a trained
model. The trained models live in `train_baseline.py` / `train_models.py` and are not wired into the CLI.

## Directory Structure

| File | Purpose |
|---|---|
| `cli.py` | Command-line entry point. Prints a text report (triage, evidence, blast radius, next steps, impact tree, affected files) or JSON. |
| `analyzer.py` | DIG loading, reverse traversal, heuristic score, triage rules (`compute_triage`), impact tree (`build_impact_tree`). Falls back to a clearly-labelled **simulated** result when no usable graph exists. |
| `evaluator.py` | Pure-Python Precision / Recall / F1 / ROC-AUC (rank-sum method, ties averaged). No third-party packages. |
| `run_eval_demo.py` | Demo of `evaluator.py` on 100 synthetic outcomes. |
| `train_baseline.py` | Ablation (always-breaking rule, version heuristic, SemVer / graph logistic regression, analyzer heuristic), 5-fold grouped CV, bootstrap CIs, results split by SemVer bump type. |
| `train_models.py`, `graph_features.py`, `gnn_model.py` | Random forest / XGBoost / structural-graph / GNN comparison (see the Person 2 section below). |
| `batch_build_digs.py` | Build pre-update DIGs for dataset samples. |
| `evaluate_localization.py` | Hit@k, Precision@k, Recall@k and MRR of the ranked impacted files against the real failing files. |
| `calibrate_triage.py` | Checks the triage tiers against labelled samples using the same code the CLI runs. |
| `test_evaluate_localization.py`, `test_triage.py` | Unit tests (58 in total). |
| `requirements.txt` | Dependencies for the training / evaluation scripts. |

---

## Getting Started / Running the Code

`cli.py`, `analyzer.py`, `evaluator.py` and `run_eval_demo.py` use only the standard library. The training and
evaluation scripts need `pip install -r eval-system/requirements.txt`. Commands below run from the repository root.

### 1. Run the risk CLI

```bash
python eval-system/cli.py \
  --repo-path . \
  --dig-path dataset/digs/quickfixj_00a7cc31784ac4a9cc27d506a73ae589d6df36d6.json \
  --dependency org.apache.mina:mina-core --old-version 2.1.5 --new-version 2.2.1
```

| Flag | Meaning |
|---|---|
| `--repo-path` (required) | Repository being analysed. |
| `--dependency` (required) | `groupId:artifactId` (an `artifactId` alone also works). |
| `--old-version`, `--new-version` (required) | Version range of the update. |
| `--dig-path` | A prebuilt DIG JSON. Without it the CLI uses `<repo>/dig.json`, then tries to build one with `graph-model/build_dig.py` (needs `javalang`), then falls back to a simulated result. |
| `--format text\|json` | Text report (default) or JSON. |
| `--require-graph` | Exit with code 2 instead of printing a simulated result. Use this in CI. |
| `--no-tree` | Omit the impact tree from the text report. |

The text report shows the risk tier and score, the **evidence** behind it (call sites found / dependency in graph
but never called / dependency not in graph) with a confidence level, the blast radius, reasoning, numbered next
steps, a plain-ASCII impact tree (`d1` = calls the dependency directly, `[test]` = test code) and the affected-files table.

JSON output keeps the original keys (`dependency`, `old_version`, `new_version`, `risk_score`, `risk_category`,
`impacted_files`, `reasoning`, `recommendation`) and adds `analysis_mode` (`graph` or `mock`), `graph_source`,
`graph_stale`, `score_category`, `triage`, `blast_radius` and `impact_tree`. **`risk_category` is now the triage
tier**, which can be higher than the tier implied by the score alone (`score_category`).

#### How triage works

The tier starts from the score (`< 0.4` SAFE, `< 0.7` RISKY, otherwise CRITICAL) and is then adjusted by evidence:

| Situation | Tier |
|---|---|
| Dependency **not found in the graph** | At least RISKY (the graph has no evidence either way; 25 of the 31 such updates in the labelled data were breaking). |
| Dependency in the graph, no call sites, MAJOR bump | At least RISKY. |
| Call sites found | Score tier. |
| No usable graph (simulated) | Labelled `[SIM]`, never presented as a real result. |

#### Things to know

- **A dependency only appears in the graph if the project's Java code calls into it.** Runtime-only libraries
  (JDBC drivers, `h2`), build plugins and BOMs never appear, so their impact cannot be measured.
- **The SAFE tier is not validated.** There are only 8 safe samples with a DIG, and 3 of the 5 samples that end up in the
  triage SAFE tier were actually breaking (`results/triage_calibration.csv`). Treat SAFE as "no evidence of impact", not as a guarantee.
  The CRITICAL boundary is clean on the data we have (19 of 19 breaking).
- **Cached graphs can go stale.** A `<repo>/dig.json` is reused as-is. The report sets `graph_stale` and prints a warning when
  `.java` files or `pom.xml` changed after it was built; delete `dig.json` to rebuild. Deleted files are not detected.
- An empty graph (e.g. a directory with no Java sources) counts as "no graph" and gives a simulated result.
- Building a DIG on demand writes `dig.json` into the target repository.

### 2. Run the metrics demo

```bash
python eval-system/run_eval_demo.py
```

### 3. Use the metrics harness from your own scripts

```python
from evaluator import compute_binary_metrics, compute_roc_auc, print_metrics_table

y_true = [...]   # 1 = the update broke the project, 0 = it did not
y_pred = [...]   # predicted probabilities from your model

m = compute_binary_metrics(y_true, y_pred, threshold=0.5)   # precision, recall, f1, tp, fp, fn, tn
auc = compute_roc_auc(y_true, y_pred)                       # returns 0.0 if only one class is present
print_metrics_table({"precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "roc_auc": auc})
```

Note that `compute_roc_auc` returns `0.0` when only one class is present (AUC is undefined). The scripts below
pool out-of-fold predictions before computing AUC, so this does not affect their results.

### 4. Baseline ablation and per-bump-type results

```bash
python eval-system/train_baseline.py                                   # 104 samples (person1/combined_dataset.csv)
python eval-system/train_baseline.py --dataset person1/final_verified_dataset.csv   # the original 59
```

5-fold `StratifiedGroupKFold` (grouped by project), pooled out-of-fold probabilities, 95% bootstrap CIs.
Writes `results/ablation_combined.csv` and `results/stratified_by_bump.csv`. Arms:

- **Always breaking (rule)**: a fixed rule, not the majority class.
- **Version heuristic**: MAJOR -> breaking, PATCH -> safe, MINOR -> a 0.5 tie. No training.
- **SemVer only** (all samples, docker-labelled samples, graph subset), **Graph only**, **SemVer + Graph**: logistic regression.
- **Analyzer heuristic**: the untrained score the CLI uses, evaluated on the graph subset.

Graph arms only cover samples that have a DIG (57), so compare arms only on the same sample subset. To build DIGs:

```bash
python eval-system/batch_build_digs.py [--limit <N>] [--safe-only]
```

### 5. Localization evaluation

```bash
python eval-system/evaluate_localization.py
```

Compares the ranked impacted files with the files that actually failed (`person1/failure_ground_truth.csv`).
Writes `results/localization_metrics.csv` (with CIs), `localization_per_sample.csv` and
`localization_unmatched_dependencies.csv` (why 17 samples have no ranking). Paths match by equality or path-segment
suffix, precision@k divides by k, and ties are broken deterministically. The report prints the sample accounting and the caveats.

### 6. Triage calibration check and tests

```bash
python eval-system/calibrate_triage.py                     # writes results/triage_calibration.csv
python -m unittest eval-system/test_evaluate_localization.py eval-system/test_triage.py
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
