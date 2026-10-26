# Results on the matched dataset (111 samples)

`person1/combined_dataset_matched.csv` = `combined_dataset.csv` (104) + 7 SAFE updates mined by
`person1/mine_matched_safe_samples.py`: **MAJOR bumps of libraries** (POM packaging jar/bundle) from the
same repositories as the BREAKING samples, accepted with the same CI checks as the other 45 CI-labelled
SAFE samples. 51 BREAKING / 60 SAFE; 109 have a DIG. SAFE MAJOR bumps go from 2 to 9.

Same protocol and scripts as `results/` (5-fold StratifiedGroupKFold by project, pooled OOF, 95%
bootstrap CIs):

```
python eval-system/train_models.py   --dataset person1/combined_dataset_matched.csv --results-dir results/matched
python eval-system/train_baseline.py --dataset person1/combined_dataset_matched.csv \
    --out results/matched/ablation.csv --strata-out results/matched/stratified_by_bump.csv
python eval-system/calibrate_triage.py --dataset person1/combined_dataset_matched.csv \
    --out results/matched/triage_calibration.csv --cv-out results/matched/triage_threshold_cv.csv
python eval-system/confound_checks.py --dataset person1/combined_dataset_matched.csv \
    --oof results/matched/oof_predictions.csv --results-dir results/matched
```

## Why only 7

The miner scanned up to 2,000 closed PRs in each of 49 repositories, found 409 bot PRs whose title is a
MAJOR library bump, and inspected 251 of them (at most 15 per repository). Most could not be accepted
(`person1/matched_safe_rejections.csv`):

- 135: no CI record on the PR head. GitHub drops old check results, and some repositories do not run
  recorded CI on bot PRs. A first run that went oldest-first (to match BUMP's dates) hit this almost
  everywhere, so the final run goes newest-first; the 7 were merged in 2025-26, not matched in age.
- 48 changed files other than `pom.xml`, 24 had an unresolvable groupId, 21 were parent POMs, BOMs or
  Jenkins plugins (packaging `pom`/`hpi`), 10 repeated an artifact already sampled, 6 had red or pending CI.

Widening to repositories without BREAKING samples would add SAFE-only projects and bring back the
project-identity shortcut that same-repo sampling removed, so it was not done.

## Risk prediction (point estimate [95% CI])

| Arm | n | ROC-AUC | MCC | Safe recall |
|---|---|---|---|---|
| Version heuristic, "major = risky" | 111 | 0.700 [0.61,0.78] | 0.281 | 0.333 |
| LR SemVer | 109 | 0.671 [0.57,0.77] | 0.171 | 0.600 |
| LR Graph | 109 | 0.642 [0.53,0.74] | 0.307 | 0.767 |
| **LR SemVer+Graph** | 109 | **0.803 [0.71,0.89]** | **0.560** | 0.767 |
| RF SemVer+Graph (best tree) | 109 | 0.811 [0.72,0.89] | 0.426 | |
| XGB SemVer+Graph | 109 | 0.787 [0.70,0.87] | 0.409 | |
| RF RepoCtx only | 109 | 0.488 [0.39,0.60] | 0.021 | |
| GNN impact-graph + SemVer | 109 | 0.789 [0.70,0.87] | 0.523 | |
| MLP SemVer (GNN control) | 109 | 0.678 [0.58,0.78] | 0.171 | |
| Analyzer heuristic (untrained) | 109 | 0.784 [0.69,0.87] | 0.371 | 0.867 |

Label-permutation p = 0.005 for LR and RF SemVer+Graph (null 95th percentile about 0.60).

## Compared with the 104-sample results

| Check | 104 samples: SemVer -> +Graph | 111 matched: SemVer -> +Graph |
|---|---|---|
| All samples with a DIG | 0.737 -> 0.841 (+0.10) | 0.671 -> 0.803 (**+0.13**) |
| Without Maven-plugin updates | 0.777 -> 0.832 (+0.05) | 0.710 -> 0.801 (**+0.09**) |
| Within-project pairs | 0.833 -> 0.922 (+0.09) | 0.769 -> 0.827 (+0.06) |
| Docker labels only (57) | 0.652 -> 0.830 (+0.18) | unchanged (no new docker samples) |

1. **SemVer was partly a dataset artifact.** Adding 7 safe MAJOR bumps drops SemVer-only from 0.74 to 0.67,
   and the "major = risky" rule from 0.76 to 0.70.
2. **The graph gain survives better.** Absolute AUC drops a little (0.84 -> 0.80), but the lift over SemVer
   grows, and without plugin updates it nearly doubles (+0.05 -> +0.09). Matching the update kind made the
   graph result *more* robust, not less.
3. **Inside MAJOR bumps the graph barely helps** (SemVer+Graph AUC 0.61, analyzer 0.63; 23 breaking vs 9
   safe). Inside MINOR it stays strong (0.81). Most of the graph signal is "how much code reaches the
   dependency" for non-major updates.
4. Models rank as before: LR, RF, XGB and the GNN are within each other's CIs; RepoCtx stays at chance.

## Triage

The cross-validated threshold check (`triage_threshold_cv.csv`) picks SAFE < 0.2 in 4 of 5 folds again,
but on this set its SAFE tier is 3 breaking of 6 (shipped: 5 of 11) and the total cost falls only from
104 to 100. On the 104-sample set the same choice looked clean (0 of 3). The evidence does not hold up,
so **the shipped thresholds (0.4 / 0.7) are unchanged**. CRITICAL is 75% breaking either way. The CLI's
SAFE tier should keep being worded as "no evidence of impact".
