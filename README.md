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
| `/dataset` | Started. `verified_samples.csv` holds 10 hand-verified BUMP examples (project, repo URL, dependency group/artifact, old/new version, failure category). All 10 are breaking updates; there are no safe-update examples yet. The GitHub crawler / mining code is not implemented. |
| `/graph-model` | Call-graph extractor done (`extract_call_graph.py`, javalang-based). It turns a cloned Java repo into an edge list of `(file, function, calls, external)` and was tested on `apache/commons-lang`. The ML model and any dependency-impact scoring are not implemented. |
| `/eval-system` | CLI (`cli.py`) and evaluation metrics (`evaluator.py`: Precision, Recall, F1, ROC-AUC) are working. `analyzer.py` is still a **mock** that returns deterministic fake scores and impacted files; it is not yet connected to `/graph-model` or `/dataset`. |
| `/docs` | Empty. |

## Quick start

Each module has its own setup and usage instructions:

- **Call-graph extractor** — see [`graph-model/README.md`](graph-model/README.md):
  `python extract_call_graph.py <path-to-cloned-repo> <output.json>`
- **CLI and evaluation harness** — see [`eval-system/README.md`](eval-system/README.md):
  `python cli.py --repo-path <path> --dependency <name> --old-version <v> --new-version <v>`
- **Dataset** — see `dataset/verified_samples.csv`

## Next steps

- Connect `eval-system/analyzer.py` to the call-graph extractor so the
  impacted files/functions come from real analysis instead of the mock.
- Extend the dataset with safe (non-breaking) updates and automate mining
  with the GitHub crawler.
- Build the risk model on features from the call graph, and evaluate it
  with `evaluator.py` against the BUMP-derived labels.

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
