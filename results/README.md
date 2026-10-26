# VerSync results and interpretation

> **See also [`matched/README.md`](matched/README.md)**: the same evaluation with 7 more SAFE updates that
> are MAJOR library bumps (111 samples). SemVer-only drops to 0.67, SemVer+Graph is 0.80, and the graph
> gain without Maven-plugin updates grows from +0.05 to +0.09. Quote those numbers alongside the ones below.

All numbers below are on `person1/combined_dataset.csv`: 104 samples (51 BREAKING, 53 SAFE), of which
**102 have a pre-update DIG** (49 BREAKING, 53 SAFE). The 2 without one: `docker-adapter` (repository
deleted) and `google-cloud-java` (57k-file monorepo, over the build timeout). 59 labels come from BUMP
docker reproduction and 45 SAFE labels from merged bot PRs with green CI (`label_source`).

Protocol everywhere: 5-fold `StratifiedGroupKFold` grouped by project, pooled out-of-fold (OOF)
probabilities, threshold 0.5, 95% bootstrap CIs (1,000 resamples).

| File | Produced by | Contents |
|---|---|---|
| `model_comparison.csv/json` | `train_models.py --dataset person1/combined_dataset.csv` | every arm, all metrics with CIs, confusion counts |
| `seed_stability.csv` | same | mean ± sd over 10 model seeds (folds fixed) |
| `label_permutation_test.json` | same | AUC p-values (200 label shuffles, full grouped CV each) |
| `oof_predictions.csv` | same | per-sample OOF P(breaking) for every arm |
| `feature_importance.csv/json/png` | same | best tree arm: built-in, OOF permutation and OOF SHAP importance |
| `ablation_combined.csv`, `stratified_by_bump.csv` | `train_baseline.py` | ablation arms and per-SemVer-bump results |
| `ablation_docker_labels.csv`, `ablation_no_maven_plugins.csv`, `confound_checks.csv` | `confound_checks.py` | the confound checks below |
| `localization_*.csv` | `evaluate_localization.py` | Hit@k, Precision@k, Recall@k, MRR; unmatched dependencies |
| `triage_calibration.csv` | `calibrate_triage.py` | SAFE / RISKY / CRITICAL tiers vs. labels |

## What changed since the previous results (57 DIGs, 8 SAFE)

- **DIGs for the 45 CI-labelled SAFE samples**, so graph arms now see 53 SAFE samples instead of 8.
- **Fixed dependency attribution** (`graph-model/`): of the 6 dependencies previously "credited to a
  sibling artifact", one was a resolver bug (`LogManager` is in `log4j-api`, not `log4j-core`). Four were
  used only through annotations (`@ParameterizedTest`) or constructors, which the extractor did not
  record; it now records `new Foo()` and annotations on imported library types. When a project declares
  several artifacts of one groupId, the call goes to the one whose name matches the package
  (`io.dropwizard.db` -> `dropwizard-db`). The updated dependency is now in **31 of the original 57 DIGs**
  (was 26), with no sample lost.
- All DIGs were rebuilt with the fixed extractor, so every graph number below differs from the old one
  for that reason too.

## Risk prediction (point estimate [95% CI], n = 102 unless stated)

| Arm | ROC-AUC | Balanced Acc | MCC | Safe recall |
|---|---|---|---|---|
| Always breaking (rule) | 0.500 | 0.500 | 0.000 | 0.000 |
| Version heuristic, "major = risky" (n=104) | 0.763 [0.68,0.84] | 0.640 | 0.327 | 0.377 |
| LR SemVer | 0.737 [0.64,0.83] | 0.716 | 0.501 | 0.962 |
| LR Graph | 0.657 [0.54,0.76] | 0.613 | 0.233 | 0.736 |
| **LR SemVer+Graph (reference)** | **0.841 [0.76,0.91]** | **0.833** | **0.666** | 0.849 |
| RF SemVer+Graph (best tree) | 0.856 [0.78,0.93] | 0.822 | 0.647 | 0.868 |
| XGB SemVer+Graph | 0.840 [0.76,0.92] | 0.793 | 0.588 | 0.830 |
| RF SemVer+Graph+Struct | 0.850 [0.77,0.92] | 0.782 | 0.568 | 0.830 |
| RF / XGB RepoCtx only (confound control) | 0.502 / 0.511 | 0.522 / 0.523 | 0.045 / 0.047 | 0.453 / 0.434 |
| GNN impact-graph + SemVer | 0.826 [0.73,0.91] | 0.822 | 0.647 | 0.868 |
| MLP SemVer (GNN control) | 0.741 [0.64,0.84] | 0.716 | 0.501 | 0.962 |
| Analyzer heuristic (untrained, as shipped) | 0.810 [0.72,0.89] | 0.669 | 0.370 | 0.868 |

Seed sd is at most 0.007 AUC for every arm. Label-permutation p = 0.005 for both LR and RF SemVer+Graph
(null 95th percentile about 0.59-0.61).

1. **Graph features now add a clear lift over SemVer: 0.74 -> 0.84 (LR).** The CIs of SemVer and
   SemVer+Graph barely overlap. The lift is concentrated in MINOR bumps, where SemVer cannot separate the
   classes: SemVer+Graph gets AUC 0.80 inside MINOR (22 breaking / 31 safe), the version rule 0.50.
2. **Tree models and the GNN tie with LR; none clearly beats it.** RF reaches 0.86, XGB 0.84, the GNN
   0.83, all within the LR CI. The GNN beats its own no-graph MLP control (0.83 vs 0.74), so the subgraph
   carries signal, but no more than LR's 7 hand-made graph features. LR stays the reference model.
3. **Repo-level context no longer predicts the label** (RepoCtx AUC 0.50). Earlier, with 8 SAFE samples
   from disjoint projects, it scored 0.80, which was dataset composition. Now that SAFE samples come from
   the same repos as BREAKING ones, that shortcut is gone.
4. **What the model uses** (RF, OOF permutation AUC drop): `is_major` 0.020, `ratio_affected_files`
   0.014, `is_patch` 0.013, `version_distance` 0.012, `maj_delta` 0.011, `direct_calls_log` 0.010. That is
   SemVer plus "how much of the project's code reaches the updated dependency".

## Is the graph gain real? Confound checks (`confound_checks.py`)

| Check | SemVer | SemVer+Graph | Gain |
|---|---|---|---|
| All 102 samples with a DIG | 0.737 | 0.841 | +0.10 |
| **Within-project pairs** (24 projects, 45 breaking/safe pairs, OOF probabilities) | 0.833 | 0.922 | +0.09 |
| **Docker labels only** (57 samples with a DIG, 8 SAFE) | 0.652 [0.41,0.86] | 0.830 [0.67,0.96] | +0.18 |
| **Without Maven-plugin updates** (88 with a DIG) | 0.777 [0.67,0.88] | 0.832 [0.74,0.92] | +0.05 |

- **Within a project the gain holds**, so it is not repo identity.
- **On docker labels only it holds**, so it is not an artifact of the CI-labelling pipeline (but with 8 SAFE
  samples the CI is wide).
- **But about half of it is the update mix.** 12 of the 45 CI-labelled SAFE samples update a Maven plugin,
  against 1 of 51 BREAKING samples. Plugins never appear in Java code, so "dependency not in the graph"
  partly encodes "this is a plugin update". Without plugin updates the gain drops to +0.05 and the CIs
  overlap. "Code that does not call the updated artifact cannot break at a call site" is a sensible
  mechanism, but the dataset over-represents it on the SAFE side.
- SemVer is still confounded too: the CI-labelled SAFE samples are mostly 2025-26 minor/patch bumps, while
  BUMP BREAKING samples are older and more often major. SemVer-only drops to 0.39-0.65 on docker labels.

**Bottom line:** graph features help, most clearly inside MINOR bumps and within projects, but the honest
size of the gain on comparable updates is about +0.05 AUC, not +0.10. To pin it down, the SAFE set needs
more library (non-plugin) updates and major bumps.

## Localization (macro average over samples)

Ground truth: files that actually failed (`person1/failure_ground_truth.csv`), mostly test files. 51
breaking samples -> 40 with code-level failures -> 39 with a DIG. Paths match by equality or path-segment
suffix.

| | All localizable (n=39) | | Dependency in DIG (n=27, was 22) | |
|---|---|---|---|---|
| Arm | Hit@10 | MRR | Hit@10 | MRR |
| Depth + fan-in ranking | 0.436 (was 0.282) | 0.319 | 0.630 | 0.460 |
| Analyzer order (as shipped) | 0.436 (was 0.282) | 0.290 | 0.630 | 0.419 |
| Random order, impacted files only | 0.368 | 0.258 | 0.532 | 0.372 |
| Random order, all DIG files | 0.239 | 0.104 | 0.252 | 0.108 |

- **5 more samples can be localized** (lithium, pgpverify-maven-plugin, license-maven-plugin, webapp,
  avans). Hit@10 over all 39 rises from 0.28 to 0.44.
- **The traversal finds the right files; ordering adds a little.** The depth + fan-in ranking beats a random
  order of the same impacted files (Hit@1 0.37 vs 0.29 on n=27), but intervals are wide.
- **12 samples still have no ranking** (`localization_unmatched_dependencies.csv`): 10 dependencies are never
  called from Java code (JDBC drivers, `h2`, `checkstyle`, servlet/validation APIs used only through
  frameworks). Of the 2 left "credited to a sibling": camunda's one call really is in `spring-beans`, and
  `google-cloud-shared-dependencies` is a BOM with no classes. Neither is a resolver bug.

## Triage tiers vs. labels (102 samples with a DIG)

| Scheme | Tier | n | Breaking | Safe |
|---|---|---|---|---|
| score only | SAFE | 70 | 25 | 45 |
| score only | RISKY | 5 | 3 | 2 |
| score only | CRITICAL | 27 | 21 | 6 |
| triage (score + evidence rules) | SAFE | 11 | 5 | 6 |
| triage | RISKY | 64 | 23 | 41 |
| triage | CRITICAL | 27 | 21 | 6 |

- **CRITICAL is no longer clean.** With 53 SAFE samples, 6 of 27 CRITICAL samples are safe (78% breaking).
  Raising the threshold to 0.8-0.9 barely changes this (79%).
- **The triage SAFE tier is unreliable:** 5 of its 11 samples break. Keep the CLI wording "no evidence of
  impact", not "safe".
- **"Dependency not in graph" is now mostly SAFE** (39 of 59), the reverse of the 57-sample data. The rule
  that never calls these SAFE is conservative (it puts them in RISKY), which is the right direction.
  The comment in `calibrate_triage.py` that calls this group "mostly breaking" describes the old data.
- The untrained analyzer score (AUC 0.81) is close to the trained LR (0.84). Its tier boundaries should be
  re-tuned on this data before they are relied on.

## Reliability flags

- 53 SAFE samples, but only 8 have docker-reproduced labels; the other 45 are CI evidence.
- Bump types are unbalanced by class: MAJOR is 23 breaking / 2 safe, PATCH 4 / 20. No per-stratum AUC
  exists for MAJOR.
- "Best tree arm" was picked by OOF AUC across 12 tree arms, which is itself optimistic. Hyperparameters
  were not tuned.
- Tree-model AUCs can move by about 0.02 across library versions (see `eval-system/requirements.txt`);
  treat differences below that as noise. The results here were produced with Python 3.13, scikit-learn,
  XGBoost and PyTorch as installed from `eval-system/requirements.txt` on macOS arm64.

## Recommendation

- Keep **LR SemVer+Graph** as the reference model (AUC 0.84, MCC 0.67 on 102 samples). Report tree models
  and the GNN as "no improvement over LR".
- Quote the graph gain as **+0.05 to +0.10 AUC**, with the plugin-mix caveat.
- Next data step: add SAFE samples that are library updates and major bumps (to match BREAKING), and
  re-tune the triage thresholds on the larger set.
