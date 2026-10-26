# VerSync: Predicting and Localizing the Impact of Dependency Updates with Dependency-Impact Graphs

*Working draft. Every number comes from a committed file in `results/` (paths given per table) and can be
regenerated with the commands in the Reproducibility section. Placeholders for prose that still needs
writing are marked TODO.*

## Abstract

Automated dependency updates (Dependabot, Renovate) are merged by the thousands, but a small share breaks the
build, and developers get little help deciding which ones to review. We present VerSync, which builds a
**Dependency-Impact Graph (DIG)** of a Java project before an update (files, functions, the library APIs
they call, and the Maven dependencies that provide them) and uses it to (1) predict whether an update will
break the build and (2) rank the files most likely to fail. On 111 real Maven updates (51 breaking, 60 safe)
from the BUMP benchmark and from CI-verified bot pull requests in the same projects, a logistic regression
on SemVer and DIG features reaches ROC-AUC 0.80 [0.71, 0.89], against 0.67 for SemVer alone, under
project-grouped cross-validation. The graph's gain survives within-project comparison and removal of
build-plugin updates, while a SemVer-only model turns out to rely largely on the composition of the sample.
When the updated dependency appears in the DIG, the true failing file is in VerSync's top 10 for 63% of
breaking updates. We also report where the approach does not work: safe-tier triage is unreliable, and
10 of 39 breaking updates involve libraries that project code never calls directly.

## 1. Introduction

TODO: motivation (volume of bot PRs, cost of breakages, SemVer is unreliable), the gap (existing tools
predict from version numbers or changelogs, not from how the project uses the library), contributions.

**Research questions.**
- **RQ1 (prediction):** Do features of how a project reaches the updated dependency improve breakage
  prediction beyond SemVer?
- **RQ2 (robustness):** Is any gain real signal, or an artifact of how the safe and breaking samples were
  collected?
- **RQ3 (localization):** Can the DIG rank the files that actually fail?
- **RQ4 (triage):** Can a three-tier SAFE / RISKY / CRITICAL output be trusted as a decision aid?

**Contributions.**
1. The DIG representation and an open-source builder for Maven projects (`graph-model/`).
2. A labelled dataset of 111 Java dependency updates with pre-update DIGs, API diffs, changelogs and
   per-sample failure locations (`person1/`, `dataset/digs/`).
3. An evaluation protocol that tests for sampling confounds explicitly (within-project pairs, label-source
   and update-kind ablations), and results showing that a widely used signal, SemVer, is largely confounded
   in this setting.
4. A file-level localization benchmark built from real build logs.

## 2. Dataset

| Source | Breaking | Safe | Label evidence |
|---|---|---|---|
| BUMP benchmark, docker-reproduced (`final_verified_dataset.csv`) | 51 | 8 | build passes before, fails after the update, in a container |
| Merged Dependabot/Renovate PRs, same projects (`safe_ci_samples.csv`) | 0 | 45 | green CI on the PR head with a real build check; only `pom.xml` changed |
| Matched safe updates (`matched_safe_samples.csv`) | 0 | 7 | as above, restricted to MAJOR bumps of libraries (jar/bundle packaging) |
| **Total** (`combined_dataset_matched.csv`) | **51** | **60** | 109 with a DIG |

- **Why safe samples come from the same projects.** The 8 BUMP safe samples come from projects with no
  breaking sample. Earlier experiments showed that repository-level properties (e.g. number of declared
  dependencies) then separate the classes on their own (AUC 0.80 for repo context alone). Mining safe
  updates from the breaking samples' own repositories removes this shortcut: repo context alone now scores
  AUC 0.49.
- **Breaking failure types** (BUMP labels): compilation 22, enforcer 15, test 12, `-Werror` 1,
  dependency lock 1.
- **Failure ground truth** (`failure_ground_truth.csv`): parsed from the reproduction logs. 40 of 51 breaking
  updates point to failing files, functions or tests; the other 11 are enforcer or lock failures that point
  only to a dependency.
- **Auxiliary data** (not used as model features in this paper): API diffs between old and new jars for all
  104 non-matched samples, changelogs/release notes (text for 83 of 104), and a CI-labelled Python set
  (25 breaking, 15 safe) for future generalization work.

## 3. Approach

### 3.1 Dependency-Impact Graph

For the project at the commit *before* the update, `build_dig.py` parses every `.java` file (javalang) and
builds:

```
FILE --CONTAINS--> FUNCTION --CALLS--> API --PROVIDED_BY--> DEPENDENCY
                   FUNCTION --CALLS--> FUNCTION   (internal calls)
```

- **API uses** are method calls, constructor calls (`new Foo()`) and annotations (`@ParameterizedTest`) on
  imported library types. Constructors and annotations matter: several libraries (JUnit params, validation,
  injection) are used almost only that way.
- **Dependency attribution** maps an API's package to a Maven artifact: a curated prefix table for libraries
  whose packages do not follow their coordinates, then the project's declared `groupId`s. When several
  declared artifacts share a `groupId`, the one whose name matches the package wins
  (`io.dropwizard.db` -> `dropwizard-db`).
- 109 of 111 samples have a DIG, all built at the intended pre-update commit (no fallback to HEAD). The 2
  missing are a deleted repository and a 57k-file monorepo.

### 3.2 Impact traversal and features

Starting from the DEPENDENCY node of the updated artifact, the analyzer walks API nodes back to their
callers and then up the internal call graph, recording each impacted function's depth. From this:

- **SemVer features (7):** major/minor/patch indicators, per-component deltas, a weighted version distance.
- **Graph features (7):** dependency found in DIG, log direct calls, log indirect calls, log impacted
  functions, log impacted files, maximum depth, share of project files impacted.
- **Structural features** (ablation only): API fan-in, PageRank share, depth distribution, modules and test
  files reached, plus repository context (size, density, declared dependencies, unresolved-call share).

### 3.3 Models

Logistic regression (reference), random forest and XGBoost (class-weighted), a small GCN over the
dependency's impact subgraph with an MLP-on-SemVer control, the untrained heuristic score shipped in the
CLI, and a fixed rule "MAJOR = breaking, PATCH = safe".

### 3.4 Localization and triage

- **Localization:** impacted files ranked by shallowest depth, then number of direct callers.
- **Triage:** SAFE / RISKY / CRITICAL from the heuristic score (thresholds 0.4 / 0.7), with evidence rules:
  an update whose dependency is absent from the DIG is never called SAFE.

## 4. Evaluation setup

- **Cross-validation:** 5-fold `StratifiedGroupKFold` grouped by project, so no project appears in both
  training and test folds. Out-of-fold probabilities are pooled across folds; threshold 0.5.
- **Uncertainty:** 95% bootstrap CIs (1,000 resamples); 10 model seeds; a label-permutation test
  (200 shuffles, full grouped CV each).
- **Confound checks** (`confound_checks.py`): pairwise AUC between breaking and safe updates *of the same
  project*; docker-labelled samples only; dataset without Maven-plugin updates.
- **Localization metrics:** Hit@k, Precision@k, Recall@k and MRR against the failing files from the logs,
  with two random baselines (shuffled impacted files; shuffled all DIG files).

## 5. Results

### RQ1: Do graph features improve prediction?

`results/matched/model_comparison.csv`, `results/matched/ablation.csv` (n = 109 with a DIG unless stated)

| Model | ROC-AUC [95% CI] | MCC |
|---|---|---|
| Rule: MAJOR = breaking (n = 111) | 0.700 [0.61, 0.78] | 0.28 |
| LR SemVer | 0.671 [0.57, 0.77] | 0.17 |
| LR Graph | 0.642 [0.53, 0.74] | 0.31 |
| **LR SemVer + Graph** | **0.803 [0.71, 0.89]** | **0.56** |
| RF SemVer + Graph | 0.811 [0.72, 0.89] | 0.43 |
| XGBoost SemVer + Graph | 0.787 [0.70, 0.87] | 0.41 |
| GCN impact subgraph + SemVer | 0.789 [0.70, 0.87] | 0.52 |
| MLP SemVer (GCN control) | 0.678 [0.58, 0.78] | 0.17 |
| Repo context only | 0.488 [0.39, 0.60] | 0.02 |
| Heuristic score (untrained, CLI) | 0.784 [0.69, 0.87] | 0.37 |

- Graph features add **+0.13 AUC** over SemVer (permutation p = 0.005). Neither group alone gets there.
- Tree models and the GCN are within the LR's CI; the GCN beats its own no-graph control by 0.11, so the
  subgraph carries signal, but no more than the 7 hand-made features.
- Most predictive (RF, OOF permutation importance): MAJOR indicator, share of files impacted, PATCH
  indicator, version distance, direct calls.
- By bump type (`results/matched/stratified_by_bump.csv`): inside MINOR updates SemVer+Graph reaches AUC
  0.81 where SemVer cannot separate the classes; inside MAJOR it is weak (0.61; 23 breaking vs 9 safe).

### RQ2: Is the gain real?

`results/confound_checks.csv`, `results/matched/confound_checks.csv`, ablation files in both folders

| Check | 104 samples: SemVer -> +Graph | 111 matched: SemVer -> +Graph |
|---|---|---|
| All samples with a DIG | 0.737 -> 0.841 (+0.10) | 0.671 -> 0.803 (+0.13) |
| Within-project pairs | 0.833 -> 0.922 (+0.09) | 0.769 -> 0.827 (+0.06) |
| Without Maven-plugin updates | 0.777 -> 0.832 (+0.05) | 0.710 -> 0.801 (+0.09) |
| Docker labels only (57 samples, 8 safe) | 0.652 -> 0.830 (+0.18) | (no new samples) |

- **The graph gain holds** within projects and on docker-reproduced labels only.
- **Update kind is a confound we found and partly fixed.** In the 104-sample set, 12 of 45 CI-labelled safe
  samples update a Maven plugin, against 1 of 51 breaking. Plugins never appear in Java code, so "dependency
  not in graph" partly meant "plugin update"; without plugins the gain halved (+0.05). Adding 7 safe MAJOR
  library updates restored it to +0.09.
- **SemVer is largely an artifact of the sample.** It scores 0.62 on docker labels only, and adding 7 safe
  MAJOR updates drops it from 0.74 to 0.67. Reported SemVer baselines on datasets whose safe samples are
  mostly minor/patch updates should be read with this in mind.

### RQ3: Can the DIG localize failures?

`results/localization_metrics.csv` (39 breaking updates with code-level failures and a DIG)

| Ranking | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---|---|---|---|
| Depth + fan-in, dependency in DIG (n = 27) | 0.370 | 0.556 | 0.630 | 0.460 |
| Random order of the same impacted files | 0.291 | 0.405 | 0.532 | 0.372 |
| Random order of all DIG files | 0.040 | 0.100 | 0.252 | 0.108 |
| Depth + fan-in, all 39 (missing dependency scores 0) | 0.256 | 0.385 | 0.436 | 0.319 |

- The traversal finds the right **set** of files (Hit@10 0.63 against 0.25 for random files); the ordering
  inside that set adds a smaller gain.
- 12 of 39 cannot be localized: in 10 the dependency is never called from Java code (JDBC drivers, `h2`,
  servlet/validation APIs used through frameworks), 1 is a BOM, 1 is attributed to a sibling artifact that
  really provides the called class.
- Fixing dependency attribution (constructors, annotations, same-group disambiguation) raised coverage from
  22 to 27 and Hit@10 over all 39 from 0.28 to 0.44.

### RQ4: Can triage be trusted?

`results/matched/triage_calibration.csv`, `results/matched/triage_threshold_cv.csv` (109 samples)

| Tier | n | Breaking | Safe |
|---|---|---|---|
| SAFE | 11 | 5 | 6 |
| RISKY | 70 | 23 | 47 |
| CRITICAL | 28 | 21 | 7 |

- CRITICAL is 75% breaking against a base rate of 45%; SAFE is not reliable (5 of 11 break).
- Choosing thresholds by grouped cross-validation did not fix SAFE consistently across the two dataset
  versions, so the shipped thresholds were kept. The CLI words SAFE as "no evidence of impact".

## 6. Threats to validity

- **Label strength.** 52 of 60 safe labels come from CI, not from a docker rebuild. CI can miss failures (no
  tests for the affected code) and can be flaky.
- **Sample size.** CIs are about ±0.09 AUC; 9 safe MAJOR updates. Differences between models are not
  significant.
- **Time mismatch.** Safe CI samples were merged in 2025-26; BUMP breaking samples are older. Repo-context
  features score at chance, which suggests code-snapshot age is not driving results, but it is not ruled out.
- **Static analysis limits.** Reflection, configuration, wildcard imports and runtime-only artifacts are
  invisible; a median of 22% of library calls per DIG resolve to no dependency ("unknown").
- **Selection.** The model-selection step (best of 12 tree arms) is optimistic; hyperparameters were not
  tuned to avoid further optimism.
- **Generalization.** Java/Maven only. A CI-labelled Python set exists (`person1/python_generalization_dataset.csv`)
  but no Python DIG builder yet.

## 7. Related work

TODO: BUMP (Reyes et al.), breaking-change studies in Maven (Raemaekers et al., Ochoa et al.),
SemVer adherence, Dependabot/Renovate empirical studies, change impact analysis, compatibility checkers
(japicmp, Revapi), GNNs on code graphs.

## 8. Future work

- **API-diff x DIG feature:** whether the project calls methods the new version removes or changes. The API
  diffs exist for all samples; this is the most direct breakage signal and is expected to help most on the
  22 compilation failures.
- More safe MAJOR updates via docker rebuilds of older bot PRs, which also gives stronger labels than CI.
- Use the trained model's probability for triage instead of the heuristic score.
- A Python DIG builder for the generalization set.

## Reproducibility

```
python eval-system/batch_build_digs.py --dataset person1/combined_dataset_matched.csv   # DIGs
python eval-system/train_models.py   --dataset person1/combined_dataset_matched.csv --results-dir results/matched
python eval-system/train_baseline.py --dataset person1/combined_dataset_matched.csv \
    --out results/matched/ablation.csv --strata-out results/matched/stratified_by_bump.csv
python eval-system/confound_checks.py --dataset person1/combined_dataset_matched.csv \
    --oof results/matched/oof_predictions.csv --results-dir results/matched
python eval-system/evaluate_localization.py
python eval-system/calibrate_triage.py --dataset person1/combined_dataset_matched.csv \
    --out results/matched/triage_calibration.csv --cv-out results/matched/triage_threshold_cv.csv
```

Library versions: `eval-system/requirements.txt`. Tree-model AUCs can move by about 0.02 across library
versions.
