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
| `/dataset` | Not started — folder scaffolded |
| `/graph-model` | Prototype call-graph extractor (`extract_call_graph.py`, javalang-based) built and tested against `apache/commons-lang`; lives on branch `Person2/Graph`, not yet merged to `main` |
| `/eval-system` | Not started — folder scaffolded |
| `/docs` | Not started — folder scaffolded |

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

Each module folder has (or will have) its own README with setup and usage
instructions specific to that part of the pipeline.
