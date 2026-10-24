# VerSync model comparison: results and interpretation

Produced by `python eval-system/train_models.py` (defaults: 1,000 bootstraps, 10 model seeds, 200
label permutations) on 57 / 59 samples with a DIG (49 BREAKING, **8 SAFE**). Protocol is identical to
`train_baseline.py`: 5-fold `StratifiedGroupKFold` grouped by project, pooled out-of-fold (OOF)
probabilities, threshold 0.5, 95% bootstrap CIs.

| File | Contents |
|---|---|
| `model_comparison.csv/json` | every arm: Precision, Recall, F1, ROC-AUC, balanced accuracy, MCC, safe-class recall + 95% CIs, confusion counts |
| `seed_stability.csv` | mean ± sd over 10 model seeds (folds fixed) |
| `label_permutation_test.json` | AUC p-values for the LR baseline and the best tree arm |
| `oof_predictions.csv` | per-sample OOF P(breaking) for every arm |
| `feature_importance.csv/json/png` | best tree arm: built-in, OOF permutation and OOF SHAP importance |

## Main table (point estimate [95% bootstrap CI])

| Arm | ROC-AUC | Balanced Acc | MCC | Safe recall | F1 (breaking) |
|---|---|---|---|---|---|
| Majority (always breaking) | 0.500 | 0.500 | 0.000 | 0.000 | 0.927 |
| LR SemVer (all 59) | 0.621 [0.39,0.82] | 0.564 | 0.089 | 0.500 | 0.736 |
| **LR SemVer+Graph (baseline)** | **0.770 [0.59,0.92]** | **0.763** | **0.399** | **0.750** | 0.854 |
| RF SemVer+Graph | 0.656 [0.43,0.85] | 0.543 | 0.079 | 0.250 | 0.854 |
| XGB SemVer+Graph | 0.608 [0.37,0.84] | 0.639 | 0.198 | 0.625 | 0.762 |
| LR SemVer+Graph+Struct | 0.783 [0.57,0.96] | 0.731 | 0.382 | 0.625 | 0.882 |
| RF SemVer+Graph+Struct | 0.725 [0.52,0.91] | 0.615 | 0.357 | 0.250 | 0.932 |
| XGB SemVer+Graph+Struct (best tree) | 0.860 [0.76,0.95] | 0.783 | 0.447 | 0.750 | 0.879 |
| XGB SemVer+Graph+**DepStruct** | 0.611 [0.38,0.84] | 0.639 | 0.198 | 0.625 | 0.762 |
| XGB **RepoCtx only** (confound control) | 0.800 [0.57,0.94] | 0.856 | 0.561 | 0.875 | 0.901 |
| GNN impact-graph + SemVer | 0.686 [0.49,0.86] | 0.732 | 0.338 | 0.750 | 0.814 |
| MLP SemVer (GNN control) | 0.619 [0.38,0.84] | 0.547 | 0.066 | 0.625 | 0.613 |

The full table with CIs on every metric is printed by the script and stored in `model_comparison.csv`.
Seed sd is ≤ 0.016 AUC for every arm, so these differences are not seed noise. However, nearly all CIs
overlap.

## What the results say

1. **RF and XGBoost do not beat the LR baseline on the same features.** On SemVer+Graph, LR reaches AUC
   0.77 / MCC 0.40, RF reaches 0.66 / 0.08 and XGB reaches 0.61 / 0.20. RF in particular collapses
   toward "always breaking" (safe recall 0.25, or 0.00 on SemVer alone) even with
   `class_weight="balanced_subsample"`: with 6–7 SAFE samples per training fold, its leaves rarely
   isolate them. This is expected for small n with only 7–14 features. A regularized linear model is
   the right baseline here.

2. **The apparent win of XGB + structural features (AUC 0.86) is driven by repo-level context, not by
   how the updated dependency is used.** Evidence:
   - `RepoCtx` alone (5 repo-level numbers: graph size, density, number of declared dependencies,
     share of unresolved calls) gets AUC 0.80 / MCC 0.56 with XGB, about the same as the best arm.
   - Replacing `Struct` with the 16 dependency-specific features (`DepStruct`: API fan-in, call share,
     PageRank share, depth distribution, modules/files reached, test-file share) drops XGB back to the
     SemVer+Graph level (0.61).
   - In the best arm, OOF permutation importance puts `declared_deps` far ahead of everything else
     (AUC drop 0.26 ± 0.07), followed by `version_distance` (0.06) and `unknown_api_share` (0.03).
     Every dependency-specific structural feature has about zero permutation importance.
     SHAP and built-in importance agree on this ranking.
   - All 8 SAFE repos declare ≤ 7 dependencies (2–7). BREAKING repos have a median of 9 (IQR 6–14).

3. **Does that make sense?** Partly. More declared dependencies does mean more surface for
   version conflicts and enforcer failures, and 15 of the 51 BREAKING labels are `ENFORCER_FAILURE`. But the
   SAFE and BREAKING samples were **mined from different sources** (SAFE: Person 1's own build
   verification pipeline; BREAKING: BUMP), from **disjoint projects**. So any repo-level property
   that differs between the two pipelines can separate the classes without saying anything about
   breakage. With 8 SAFE samples we cannot tell these two explanations apart. **Treat the 0.86 as an
   upper bound inflated by dataset composition, not as evidence that structural graph features
   work.**

4. **What drives the LR baseline's lift (0.62 → 0.77).** In a full-data fit of the standardized
   model, the largest coefficient is `is_patch` (−1.08: patch bumps lean SAFE). The graph contribution
   comes from `direct_calls_log` (+0.80), `affected_files_log` (+0.65) and `dep_found_in_graph` (+0.41).
   Neither SemVer alone (0.65) nor Graph alone (0.61) gets there; the combination does. 6 of the 8 SAFE updates are build plugins, BOMs or runtime-only artifacts
   (enforcer-rules, checkstyle, jersey-hk2, the shared-dependencies BOM, …) that the code never calls
   (found in the DIG for 25% of SAFE vs 49% of BREAKING samples; mean direct-call log 0.22 vs 1.18).
   "Code never calls the updated artifact, so call sites cannot break" is a sensible mechanism. It is
   also partly a property of what kind of update each mining source tends to contain.

5. **GNN.** The tiny GCN over the dependency's impact subgraph (AUC 0.686, MCC 0.34) beats its own
   no-graph MLP control (0.619 / 0.07), so the subgraph carries some signal. But it is **below the
   LR SemVer+Graph baseline** (0.770 / 0.40), whose 7 hand-made graph features capture the same "is the
   dependency used, and how much" information more efficiently. It does not catastrophically overfit:
   fixed epochs, weight decay and 3k parameters keep OOF performance stable across seeds
   (sd 0.003 AUC). It simply does not help beyond the baseline. For 31 of 57 samples the dependency is
   not in the DIG, so the GNN sees a single empty node and falls back to SemVer.

6. **Label-permutation test** (200 shuffles, full grouped CV each): LR SemVer+Graph p = 0.025, best XGB
   p = 0.005. Both models pick up *something* beyond chance. Given point 2, for XGB that "something"
   is mostly repo identity.

## Reliability flags (8 SAFE samples)

- Safe-class recall moves in steps of **0.125** (one sample). The difference between 0.625 and 0.750 is
  one repository.
- Bootstrap CIs on safe recall are typically [0.25, 1.00], and on MCC roughly ±0.3. **No pairwise
  difference between non-trivial arms is statistically reliable.** The CIs of the LR baseline and the
  best XGB arm overlap almost completely.
- "Best tree arm" was chosen by OOF AUC across 12 tree arms. That selection step is itself optimistic.
- html2pop3 and micro-migration each contribute 2 of the 8 SAFE samples. Grouping keeps them in one
  fold, but they make up 50% of the SAFE class.
- Hyperparameters were not tuned. Tuning on these folds would make the numbers look better and mean
  less.

## Recommendation

- Keep **LR SemVer+Graph** as the reference model. Report RF/XGB as "no improvement at n = 57".
- Do **not** ship `declared_deps` or other `RepoCtx` features in a deployed model until SAFE samples
  come from the same source (and ideally the same repos) as BREAKING ones.
- The structural/GNN arms are worth re-running once there are more SAFE samples and the
  dependency-to-DIG matching improves (currently 26 / 57; see `dataset/DIG_BUILD_REPORT.md`).

---

# Paper 2 evaluation: SemVer strata, localization and triage

Produced by `train_baseline.py`, `evaluate_localization.py` and `calibrate_triage.py` (all in `eval-system/`).

| File | Contents |
|---|---|
| `ablation_combined.csv` | Ablation arms on the 104-sample combined dataset (graph arms use the 57 samples with a DIG), with 95% CIs |
| `stratified_by_bump.csv` | Every arm split by SemVer bump type (counts, accuracy + CI, recalls, AUC when both classes have >= 3 samples) |
| `localization_metrics.csv`, `localization_per_sample.csv` | Hit@k, Precision@k, Recall@k, MRR with bootstrap CIs; one row per sample |
| `localization_unmatched_dependencies.csv` | Why 17 samples have no ranking at all |
| `triage_calibration.csv` | SAFE / RISKY / CRITICAL tiers vs. true labels on the DIG samples |

## Ablation (point estimate [95% bootstrap CI])

| Arm | n | ROC-AUC | Balanced Acc | Safe recall |
|---|---|---|---|---|
| Always breaking (rule) | 104 | 0.500 [0.50, 0.50] | 0.50 | 0.00 |
| Version heuristic (all 104) | 104 | 0.763 [0.68, 0.84] | 0.64 | 0.38 |
| Version heuristic (docker labels n=59) | 59 | 0.723 [0.53, 0.89] | 0.64 | 0.38 |
| Version heuristic (subset n=57) | 57 | 0.736 [0.55, 0.89] | 0.65 | 0.38 |
| SemVer only (all 104) | 104 | 0.736 [0.64, 0.82] | 0.71 | 0.96 |
| SemVer only (docker labels n=59) | 59 | 0.621 [0.39, 0.82] | 0.56 | 0.50 |
| SemVer only (subset n=57) | 57 | 0.652 [0.41, 0.86] | 0.58 | 0.62 |
| Graph only (subset n=57) | 57 | 0.606 [0.37, 0.83] | 0.57 | 0.75 |
| SemVer + Graph (n=57) | 57 | 0.770 [0.59, 0.92] | 0.76 | 0.75 |
| Analyzer heuristic (subset n=57) | 57 | 0.801 [0.64, 0.93] | 0.69 | 1.00 |

1. **The SemVer-only gain on 104 samples appears to come from the sample mix, not real signal.** The same model on the 59 docker-labelled samples gets AUC 0.62. The 45 added
   CI-labelled safe samples are almost all minor/patch bumps from recent years, while the breaking ones are older and more often major.
2. **A fixed version rule is competitive.** "Major = risky" scores AUC 0.74 on the graph subset and 0.76 on all 104, within the CIs of every trained arm.
   Within MINOR and PATCH it cannot discriminate (AUC exactly 0.5), so its score comes entirely from the mix of bump types.
3. **The graph arms still use only 57 samples**: the 45 CI-labelled safe samples have no DIG.
4. Per-bump-type cells are small (e.g. a handful of safe samples in MINOR and PATCH), so the stratified numbers are descriptive only.

## Localization (macro average over samples)

Ground truth: files that actually failed (`person1/failure_ground_truth.csv`), mostly test files. 51 breaking samples -> 40 with code-level
failures (11 fail at the dependency level and are excluded) -> 39 with a built DIG. Paths match by equality or path-segment suffix.

**All localizable samples (n=39)** (samples whose dependency is absent from the DIG get an empty ranking and score 0)

| Arm | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---|---|---|---|
| Depth + fan-in ranking | 0.128 | 0.205 | 0.282 | 0.179 |
| Analyzer order (as shipped) | 0.154 | 0.205 | 0.282 | 0.195 |
| Random order, impacted files only | 0.117 | 0.190 | 0.262 | 0.165 |
| Random order, all DIG files | 0.040 | 0.100 | 0.239 | 0.104 |

**Dependency matched in DIG (n=22)**

| Arm | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---|---|---|---|
| Depth + fan-in ranking | 0.227 | 0.364 | 0.500 | 0.318 |
| Analyzer order (as shipped) | 0.273 | 0.364 | 0.500 | 0.346 |
| Random order, impacted files only | 0.208 | 0.336 | 0.464 | 0.293 |
| Random order, all DIG files | 0.041 | 0.099 | 0.243 | 0.105 |

- **The traversal finds the right set of files; the ordering adds little.** Randomly shuffling the impacted files performs about the same.
  With a median of 5 files per ranking there is little to order. Intervals are wide (n = 22).
- **17 of 39 samples cannot be localized because the updated dependency is not in the DIG.** 11 are never called from the project's Java code
  (e.g. JDBC drivers, `h2`, `checkstyle`): a limit of static call graphs. 6 are calls the resolver credits to a different artifact of the same group
  (e.g. `dropwizard-client` calls attributed to `dropwizard-core`; all `junit-jupiter` calls to `junit-jupiter-api`). A quick offline check that
  matched those 6 by groupId gave a ranking for all of them and a top-1 hit for 3 of 6. That is too few samples to act on, but it suggests fixing
  the attribution in `graph-model/dependency_resolver.py`.
- Ground-truth files are mostly test files taken from test/compile logs, which favours code that tests exercise.

## Triage tiers vs. true labels (57 samples with a DIG)

| Scheme | Tier | n | Breaking | Safe |
|---|---|---|---|---|
| score only tier | SAFE | 36 | 28 | 8 |
| score only tier | RISKY | 2 | 2 | 0 |
| score only tier | CRITICAL | 19 | 19 | 0 |
| triage tier | SAFE | 5 | 3 | 2 |
| triage tier | RISKY | 33 | 27 | 6 |
| triage tier | CRITICAL | 19 | 19 | 0 |

- **CRITICAL is clean** on this data (every sample at or above 0.7 was breaking); the score-only SAFE tier is not (most of its samples were breaking).
  The evidence rules therefore never call a dependency that is missing from the graph SAFE.
- **The SAFE tier cannot be validated**: with 8 safe samples, 3 of the 5 in the triage SAFE tier were breaking. The CLI words SAFE as "no evidence of impact", not as a guarantee.
- Tier boundaries were sanity-checked, not statistically calibrated.

## Reproducibility notes

- Deterministic arms (always-breaking rule, version heuristic, logistic regression) reproduce exactly across machines.
- `model_comparison.csv` was produced with the library versions in `eval-system/requirements.txt`. Re-running `train_models.py --no-gnn` with
  Python 3.12.3, scikit-learn 1.8.0, XGBoost 3.4.1 and NumPy 2.4.4 gave identical results to the original code on `main` in that environment, but RF and
  XGBoost AUCs moved by up to about 0.016 compared with the committed files, similar to the seed standard deviation. Record library versions when quoting these numbers
  and treat tree-model differences below about 0.02 AUC as noise.
- The localization and triage results depend on `analyzer.py` traversing the DIG in a fixed order; this was made deterministic (`sorted(target_apis)`), and the outputs are
  byte-identical across Python hash seeds.
