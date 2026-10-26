# VerSync

AI-Based Dependency Update Risk Prediction and Impact Localization in Software Repositories

## Overview

Updating a dependency in a Java project is a gamble: it might be a safe
patch, or it might silently break something downstream. VerSync aims to
predict, before you merge an update, **how risky it is** and **which parts
of the codebase are likely to be affected**, by combining:

- historical evidence of real dependency-update breakages,
- a static picture of how the project actually calls into that dependency
  (and into itself), and
- a way to measure whether the predictions are actually useful.

## Pipeline

The three modules form a loose pipeline, though each can be developed and
tested independently:

1. **`/dataset`** mines real-world examples of dependency updates that broke
   (or didn't break) a build — using the [BUMP dataset](https://github.com/chains-project/bump)
   and/or a custom GitHub crawler — to produce labeled training data.
2. **`/graph-model`** parses a Java repo into a method-level call graph
   (internal calls vs. calls into external libraries), and uses that
   structure — combined with the labeled data from `/dataset` — to train
   models that predict update risk and localize likely impact.
3. **`/eval-system`** wraps the above into a CLI tool a developer could
   actually run against a repo + a proposed dependency bump, and provides
   an evaluation harness to score prediction quality against held-out BUMP
   examples.

## Structure

- `/dataset` — **Person 1**: BUMP dataset mining, GitHub crawler for labeled
  dependency-update examples
- `/graph-model` — **Person 2**: Java call-graph extraction + ML models for
  risk prediction and impact localization
- `/eval-system` — **Person 3**: CLI tool + evaluation harness
- `/docs` — shared notes, paper drafts, design discussions

## Status

| Module | Status |
|---|---|
| `/dataset` (mined data lives in `person1/`) | 59 verified BUMP samples (51 breaking, 8 safe) in `person1/final_verified_dataset.csv`, plus 45 safe updates mined from merged Dependabot/Renovate PRs with passing CI. Combined in `person1/combined_dataset.csv`: 104 samples (51 breaking, 53 safe); the `label_source` column separates docker-reproduced labels from GitHub-CI labels. Also: failure ground truth (`failure_ground_truth.csv`; 40 of the 51 verified breaking samples point to failing files), API diffs, changelogs, and a Python generalization set (25 breaking, 15 safe, 11 unverified). |
| `/graph-model` | Java call-graph extractor (method calls, constructor calls and annotations on library types), Maven dependency resolver and Dependency-Impact-Graph (DIG) builder (`build_dig.py`). 102 of the 104 combined samples have a pre-update DIG in `dataset/digs/`; 2 could not be built (see `dataset/DIG_BUILD_REPORT.md`). |
| `/eval-system` | `analyzer.py` traverses the real DIG (no longer a mock). The CLI prints a SAFE / RISKY / CRITICAL triage with next steps, a blast-radius summary and an ASCII impact tree, and labels results as simulated when no usable graph exists. Evaluation: baseline ablation with bootstrap CIs, SemVer-stratified results and a version-heuristic baseline, RF / XGBoost / GNN comparison (`train_models.py`), localization metrics (Hit@k, Precision@k, Recall@k, MRR), a triage calibration check and confound checks (`confound_checks.py`). 58 unit tests. Outputs are in `/results`. |
| `/docs` | Empty. |

Headline result so far (`results/README.md`): a logistic regression on SemVer + graph features is the
reference model (ROC-AUC 0.84, 95% CI 0.76-0.91, MCC 0.67, n = 102 samples with a DIG, 53 of them safe).
Graph features lift SemVer from 0.74 to 0.84, and the lift holds within projects and on docker labels
only. About half of it comes from safe samples over-representing Maven-plugin updates (without them:
0.78 -> 0.83). Tree models and a small GNN tie with LR.

## Quick start

Each module has its own setup and usage instructions:

- **Call-graph extractor** - see [`graph-model/README.md`](graph-model/README.md):
  `python extract_call_graph.py <path-to-cloned-repo> <output.json>`
- **CLI** - see [`eval-system/README.md`](eval-system/README.md):
  `python eval-system/cli.py --repo-path <path> --dependency <group:artifact> --old-version <v> --new-version <v> [--dig-path <dig.json>]`
  (add `--require-graph` in CI so a simulated result can never pass for a real one)
- **Evaluation** - `python eval-system/train_baseline.py` (ablation and per-bump-type results),
  `python eval-system/evaluate_localization.py` (localization metrics),
  `python eval-system/calibrate_triage.py` (triage tiers vs labels),
  `python eval-system/train_models.py --dataset person1/combined_dataset.csv` (model comparison),
  `python eval-system/confound_checks.py` (is the graph gain real?)
- **Dataset** - see `person1/README.md`, `person1/final_verified_dataset.csv` and `person1/combined_dataset.csv`

## Next steps

- **Collect safe samples that match the breaking ones**: library (not Maven-plugin) updates and major
  bumps. 12 of the 45 CI-labelled safe samples are plugin updates (vs 1 of 51 breaking), and SemVer alone
  drops to AUC 0.62 on the docker-labelled samples, so both feature groups are partly confounded.
- **Re-tune the triage tiers on the 102-sample set.** With 53 safe samples, 6 of 27 CRITICAL samples are
  safe and 5 of 11 SAFE-tier samples break (`results/triage_calibration.csv`).
- 12 of the 39 localizable breaking samples still have no ranking: 10 dependencies are never called from
  Java code (a limit of static call graphs), 1 is a BOM, 1 is correctly attributed to a sibling artifact.

## Team workflow

- Each person works on a branch named `Person<N>/<Topic>` (e.g.
  `Person2/Graph`), branched from an up-to-date `main`.
- `main` stays stable: work is pushed to a personal branch and opened as a
  pull request rather than pushed/merged directly, except for
  project-wide docs (like this README) that don't touch any module's code.
- Pull `main` before starting new work to avoid diverging too far.

## Getting started

```bash
git clone https://github.com/Sanjanavadari/VerSync.git
cd VerSync
git checkout -b Person<N>/<Topic>
```
