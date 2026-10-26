"""
train_models.py

VerSync tree-model, structural-graph and GNN arms, evaluated under the exact
protocol of train_baseline.py so every row is directly comparable with the
logistic-regression baseline.

Reused from train_baseline.py (not re-implemented):
  * load_and_featurize_dataset / find_dig_for_sample  - SemVer + graph features
  * evaluate_model_pipeline                           - the LR baseline itself
  * evaluate_majority_baseline
  * compute_extended_metrics / compute_bootstrap_ci   - metrics + 95% bootstrap CIs
  * format_row                                        - ablation table layout
  * StratifiedGroupKFold(n_splits=5), grouped by project, pooled out-of-fold
    probabilities, threshold 0.5

New arms:
  * Random Forest (class_weight="balanced_subsample")
  * XGBoost (scale_pos_weight = n_safe / n_breaking per training fold)
  * Structural DIG features (graph_features.py) fed to LR / RF / XGB
  * A tiny GCN on the dependency's impact subgraph (gnn_model.py) plus an
    identical network without the graph branch as its control

Hyperparameters are fixed a priori (no tuning) because there is no data to
spare for a nested tuning loop: with 8 safe samples any tuning on the CV
folds would inflate the reported numbers.

Outputs (in --results-dir, default results/):
  model_comparison.csv / .json   - one row per arm, metrics + 95% CIs
  seed_stability.csv             - mean/sd of AUC, MCC, balanced acc over seeds
  label_permutation_test.json    - AUC p-values for the baseline and best tree arm
  oof_predictions.csv            - pooled out-of-fold P(breaking) per sample/arm
  feature_importance.csv / .json - built-in, OOF permutation and OOF SHAP importance
  feature_importance.png         - top features of the best tree arm

Usage:
    python eval-system/train_models.py [--bootstrap-iters 1000] [--seeds 10]
        [--perm-iters 200] [--no-gnn] [--results-dir results]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import (
    compute_bootstrap_ci,
    compute_extended_metrics,
    evaluate_majority_baseline,
    evaluate_model_pipeline,
    find_dig_for_sample,
    format_row,
    load_and_featurize_dataset,
)
from graph_features import DEP_STRUCT_COLS, REPO_CONTEXT_COLS, STRUCTURAL_COLS, extract_structural_features

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))

# Same column lists as train_baseline.run_experiment (defined locally there).
SEMVER_COLS = ["is_major", "is_minor", "is_patch", "maj_delta", "min_delta", "pat_delta", "version_distance"]
GRAPH_COLS = [
    "dep_found_in_graph",
    "direct_calls_log",
    "indirect_calls_log",
    "affected_functions_log",
    "affected_files_log",
    "max_depth",
    "ratio_affected_files",
]
METRIC_KEYS = ["precision", "recall", "f1", "roc_auc", "balanced_acc", "mcc", "safe_recall"]
N_SPLITS = 5
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Models (fixed hyperparameters, chosen a priori for n~59)
# ---------------------------------------------------------------------------

def make_lr(y_train: np.ndarray, seed: int) -> Pipeline:
    # Identical to train_baseline.evaluate_model_pipeline
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)),
    ])


def make_rf(y_train: np.ndarray, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def make_xgb(y_train: np.ndarray, seed: int) -> XGBClassifier:
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    return XGBClassifier(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        # Positive class = BREAKING (majority): down-weight it to balance safe samples.
        scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
    )


MODEL_FACTORIES: Dict[str, Callable[[np.ndarray, int], Any]] = {"LR": make_lr, "RF": make_rf, "XGB": make_xgb}


# ---------------------------------------------------------------------------
# Shared CV protocol
# ---------------------------------------------------------------------------

def cv_oof_probs(
    model_kind: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, List[Tuple[Any, np.ndarray]]]:
    """
    Same protocol as train_baseline.evaluate_model_pipeline (StratifiedGroupKFold,
    5 splits, no shuffling -> identical folds), generalized to any model.
    Returns pooled out-of-fold P(breaking) and the (fitted model, val_idx) per fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y), dtype=float)
    fold_models = []
    for train_idx, val_idx in sgkf.split(X, y, groups):
        model = MODEL_FACTORIES[model_kind](y[train_idx], seed)
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[val_idx])
        classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
        if probs.shape[1] == 2:
            oof[val_idx] = probs[:, 1]
        else:
            oof[val_idx] = 1.0 if classes[0] == 1 else 0.0
        fold_models.append((model, val_idx))
    return oof, fold_models


def bootstrap_cis(y: np.ndarray, probs: np.ndarray, iters: int) -> Dict[str, Tuple[float, float]]:
    return {k: compute_bootstrap_ci(y, probs, k, n_iters=iters) for k in METRIC_KEYS}


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

def load_all_features(dataset: str, dig_dir: Optional[str]) -> Dict[str, Any]:
    rows, y, groups, feat_dicts = load_and_featurize_dataset(dataset, dig_dir)

    struct_dicts, provenance, deps = [], [], []
    for r, fd in zip(rows, feat_dicts):
        dep = f"{r['dependency_group_id']}:{r['dependency_artifact_id']}"
        deps.append(dep)
        dig = (find_dig_for_sample(r["project"], r.get("source_json", ""), dig_dir, r.get("pre_commit_sha", ""))
               if fd["has_graph"] else None)
        struct_dicts.append(extract_structural_features(dig, dep))
        meta = (dig or {}).get("build_metadata", {})
        provenance.append(meta.get("checkout_mode", "unknown" if dig else "no_dig"))

    def mat(dicts: List[Dict[str, float]], cols: List[str]) -> np.ndarray:
        return np.array([[d[c] for c in cols] for d in dicts], dtype=float)

    return {
        "rows": rows,
        "y": y,
        "groups": groups,
        "deps": deps,
        "provenance": provenance,
        "has_graph": np.array([fd["has_graph"] == 1.0 for fd in feat_dicts]),
        "dep_found": np.array([fd["dep_found_in_graph"] == 1.0 for fd in feat_dicts]),
        "X": {
            "SemVer": mat(feat_dicts, SEMVER_COLS),
            "Graph": mat(feat_dicts, GRAPH_COLS),
            "Struct": mat(struct_dicts, STRUCTURAL_COLS),
            "DepStruct": mat(struct_dicts, DEP_STRUCT_COLS),
            "RepoCtx": mat(struct_dicts, REPO_CONTEXT_COLS),
        },
    }


FEATURE_SETS = {
    "SemVer": ["SemVer"],
    "Graph": ["Graph"],
    "Struct": ["Struct"],
    "DepStruct": ["DepStruct"],
    "RepoCtx": ["RepoCtx"],
    "SemVer+Graph": ["SemVer", "Graph"],
    "SemVer+Graph+DepStruct": ["SemVer", "Graph", "DepStruct"],
    "SemVer+Graph+Struct": ["SemVer", "Graph", "Struct"],
}


def feature_names(fs: str) -> List[str]:
    cols = {"SemVer": SEMVER_COLS, "Graph": GRAPH_COLS, "Struct": STRUCTURAL_COLS,
            "DepStruct": DEP_STRUCT_COLS, "RepoCtx": REPO_CONTEXT_COLS}
    return [c for block in FEATURE_SETS[fs] for c in cols[block]]


def feature_matrix(data: Dict[str, Any], fs: str, idx: np.ndarray) -> np.ndarray:
    return np.hstack([data["X"][block][idx] for block in FEATURE_SETS[fs]])


# ---------------------------------------------------------------------------
# Feature importance (best tree arm)
# ---------------------------------------------------------------------------

def oof_permutation_importance(
    fold_models: List[Tuple[Any, np.ndarray]],
    X: np.ndarray,
    y: np.ndarray,
    base_oof: np.ndarray,
    n_repeats: int = 30,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permutation importance on held-out data: for each feature, shuffle it within
    every validation fold, re-predict with that fold's model, pool the OOF
    probabilities and measure the drop in pooled ROC-AUC. Returns (mean, sd).
    """
    rng = np.random.RandomState(seed)
    base_auc = compute_extended_metrics(y, base_oof)["roc_auc"]
    n_feat = X.shape[1]
    drops = np.zeros((n_feat, n_repeats))
    for j in range(n_feat):
        for r in range(n_repeats):
            perm_oof = np.zeros(len(y))
            for model, val_idx in fold_models:
                Xv = X[val_idx].copy()
                Xv[:, j] = Xv[rng.permutation(len(val_idx)), j]
                perm_oof[val_idx] = model.predict_proba(Xv)[:, 1]
            drops[j, r] = base_auc - compute_extended_metrics(y, perm_oof)["roc_auc"]
    return drops.mean(1), drops.std(1)


def oof_shap_importance(fold_models: List[Tuple[Any, np.ndarray]], X: np.ndarray) -> Optional[np.ndarray]:
    """Mean |SHAP| of P(breaking) computed with each fold's model on its own validation rows."""
    try:
        import shap
    except ImportError:
        return None
    vals = np.zeros_like(X, dtype=float)
    for model, val_idx in fold_models:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X[val_idx])
        if isinstance(sv, list):  # older shap: one array per class
            sv = sv[1]
        elif sv.ndim == 3:  # (n, features, classes)
            sv = sv[:, :, 1]
        vals[val_idx] = sv
    return np.abs(vals).mean(0)


def plot_importance(names: List[str], perm_mean: np.ndarray, perm_sd: np.ndarray,
                    builtin: np.ndarray, title: str, path: str, top_k: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(perm_mean)[::-1][:top_k][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * top_k + 1.6), sharey=True)
    ylab = [names[i] for i in order]
    ypos = np.arange(len(order))

    axes[0].barh(ypos, perm_mean[order], xerr=perm_sd[order], color="#3b6ea8",
                 error_kw={"ecolor": "#8a8f98", "capsize": 2, "lw": 1})
    axes[0].axvline(0, color="#8a8f98", lw=0.8)
    axes[0].set_title("OOF permutation importance\n(drop in pooled ROC-AUC)", fontsize=10)
    axes[0].set_yticks(ypos, ylab, fontsize=9)

    axes[1].barh(ypos, builtin[order], color="#3b6ea8")
    axes[1].set_title("Built-in importance\n(mean over fold models)", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#e3e5e8", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Label-permutation test
# ---------------------------------------------------------------------------

def label_permutation_pvalue(
    model_kind: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    observed_auc: float, n_iters: int, seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """
    p-value for 'pooled OOF ROC-AUC this high by chance': shuffle labels across
    samples (folds are re-derived from the shuffled labels, groups preserved),
    re-run the full CV, and count permuted AUCs >= observed.
    """
    rng = np.random.RandomState(seed)
    null = []
    for _ in range(n_iters):
        y_perm = y[rng.permutation(len(y))]
        oof, _ = cv_oof_probs(model_kind, X, y_perm, groups, seed=seed)
        null.append(compute_extended_metrics(y_perm, oof)["roc_auc"])
    null = np.array(null)
    return {
        "observed_auc": round(float(observed_auc), 4),
        "null_mean_auc": round(float(null.mean()), 4),
        "null_p95_auc": round(float(np.percentile(null, 95)), 4),
        "p_value": round(float((1 + np.sum(null >= observed_auc)) / (1 + n_iters)), 4),
        "n_iters": n_iters,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_all_features(args.dataset, args.dig_dir)
    y, groups = data["y"], data["groups"]
    n_total = len(y)
    all_idx = np.arange(n_total)
    sub_idx = np.where(data["has_graph"])[0]
    n_sub = len(sub_idx)
    y_sub, g_sub = y[sub_idx], groups[sub_idx]

    prov_counts: Dict[str, int] = {}
    for i in sub_idx:
        prov_counts[data["provenance"][i]] = prov_counts.get(data["provenance"][i], 0) + 1

    print(f"\n[VerSync] Loaded {n_total} samples ({int(y.sum())} BREAKING, {int((y == 0).sum())} SAFE), "
          f"{len(set(groups))} project groups")
    print(f"[VerSync] Samples with a DIG: {n_sub} ({int(y_sub.sum())} BREAKING, {int((y_sub == 0).sum())} SAFE); "
          f"DIG checkout provenance: {prov_counts}")
    print(f"[VerSync] Updated dependency found in its DIG: {int(data['dep_found'][sub_idx].sum())} / {n_sub}")

    results: List[Dict[str, Any]] = []
    oof_table: Dict[str, np.ndarray] = {}
    fold_store: Dict[str, Tuple[List, np.ndarray, str, str]] = {}

    def record(name: str, kind: str, fs: str, idx: np.ndarray, probs: np.ndarray, seed_stochastic: bool) -> None:
        yy = y[idx]
        m = compute_extended_metrics(yy, probs)
        cis = bootstrap_cis(yy, probs, args.bootstrap_iters)
        results.append({"arm": name, "model": kind, "features": fs, "n": len(idx),
                        "n_safe": int((yy == 0).sum()), "metrics": m, "ci95": cis,
                        "stochastic": seed_stochastic})
        full = np.full(n_total, np.nan)
        full[idx] = probs
        oof_table[name] = full

    # Arm 1: majority baseline, Arm 2: LR SemVer on all samples (as in train_baseline)
    maj_probs, _ = evaluate_majority_baseline(y)
    record("Majority (always breaking)", "Majority", "-", all_idx, maj_probs, False)
    lr_all, _ = evaluate_model_pipeline(data["X"]["SemVer"], y, groups)
    record(f"LR SemVer (all {n_total})", "LR", "SemVer", all_idx, lr_all, False)
    if n_sub < n_total:
        for kind in ("RF", "XGB"):
            p, _ = cv_oof_probs(kind, data["X"]["SemVer"], y, groups)
            record(f"{kind} SemVer (all {n_total})", kind, "SemVer", all_idx, p, True)

    # Graph-subset arms
    tag = f"n={n_sub}"
    arms: List[Tuple[str, str]] = [
        ("LR", "SemVer"), ("LR", "Graph"), ("LR", "SemVer+Graph"), ("LR", "SemVer+Graph+Struct"),
        ("RF", "SemVer"), ("RF", "SemVer+Graph"), ("RF", "Struct"), ("RF", "SemVer+Graph+Struct"),
        ("XGB", "SemVer"), ("XGB", "SemVer+Graph"), ("XGB", "Struct"), ("XGB", "SemVer+Graph+Struct"),
        # Confound controls: dependency-specific structure vs repo-level context only
        ("RF", "SemVer+Graph+DepStruct"), ("XGB", "SemVer+Graph+DepStruct"),
        ("RF", "RepoCtx"), ("XGB", "RepoCtx"),
    ]
    for kind, fs in arms:
        X = feature_matrix(data, fs, sub_idx)
        if kind == "LR":
            p, _ = evaluate_model_pipeline(X, y_sub, g_sub)
            folds = None
        else:
            p, folds = cv_oof_probs(kind, X, y_sub, g_sub)
        short = fs.replace("SemVer+Graph+", "SV+G+")
        name = f"{kind} {short} ({tag})" if len(f"{kind} {short} ({tag})") <= 30 else f"{kind} {short}"
        record(name, kind, fs, sub_idx, p, kind != "LR")
        if folds is not None:
            fold_store[name] = (folds, X, kind, fs)

    # GNN arms
    gnn_graphs = None
    if not args.no_gnn:
        from gnn_model import build_impact_subgraph, gnn_oof_probs
        print("[VerSync] Building impact subgraphs for the GNN arm...")
        gnn_graphs = []
        for i in sub_idx:
            r = data["rows"][i]
            dig = find_dig_for_sample(r["project"], r.get("source_json", ""), args.dig_dir,
                                      r.get("pre_commit_sha", ""))
            gnn_graphs.append(build_impact_subgraph(dig, data["deps"][i]))
        X_sv = data["X"]["SemVer"][sub_idx]
        p = gnn_oof_probs(gnn_graphs, X_sv, y_sub, g_sub, seed=DEFAULT_SEED, use_graph=True)
        record(f"GNN impact-graph+SemVer", "GNN", "Subgraph+SemVer", sub_idx, p, True)
        p = gnn_oof_probs(gnn_graphs, X_sv, y_sub, g_sub, seed=DEFAULT_SEED, use_graph=False)
        record(f"MLP SemVer (GNN control)", "MLP", "SemVer", sub_idx, p, True)

    # ---- Table (same layout as train_baseline) ----
    header = (
        f"| {'Model Arm'.ljust(30)} | {'Precision'.ljust(15)} | {'Recall'.ljust(15)} | {'F1'.ljust(15)} "
        f"| {'ROC-AUC'.ljust(15)} | {'Balanced Acc'.ljust(15)} | {'MCC'.ljust(15)} | {'Safe-class Recall'.ljust(17)} |"
    )
    sep = f"|{'-' * 32}|{'-' * 17}|{'-' * 17}|{'-' * 17}|{'-' * 17}|{'-' * 17}|{'-' * 17}|{'-' * 19}|"
    print("\n" + "=" * 140)
    print("                    VERSYNC MODEL COMPARISON (LR baseline vs RF vs XGBoost vs graph arms)")
    print("=" * 140)
    print(header)
    print(sep)
    for r in results:
        print(format_row(r["arm"], r["metrics"], r["ci95"]))
    print("=" * 140)

    # ---- Seed stability for stochastic arms ----
    stability = []
    if args.seeds > 1:
        print(f"\n[VerSync] Seed stability over {args.seeds} seeds (folds fixed, model seed varies):")
        for r in results:
            if not r["stochastic"]:
                continue
            aucs, mccs, baccs = [], [], []
            idx = sub_idx if r["n"] == n_sub else all_idx
            seeds = range(args.seeds) if r["model"] in ("RF", "XGB") else range(min(args.seeds, args.gnn_seeds))
            for s in seeds:
                if r["model"] in ("RF", "XGB"):
                    X = feature_matrix(data, r["features"], idx)
                    p, _ = cv_oof_probs(r["model"], X, y[idx], groups[idx], seed=s)
                else:
                    p = gnn_oof_probs(gnn_graphs, data["X"]["SemVer"][sub_idx], y_sub, g_sub,
                                      seed=s, use_graph=(r["model"] == "GNN"))
                m = compute_extended_metrics(y[idx], p)
                aucs.append(m["roc_auc"]); mccs.append(m["mcc"]); baccs.append(m["balanced_acc"])
            row = {"arm": r["arm"], "n_seeds": len(aucs),
                   "auc_mean": round(float(np.mean(aucs)), 3), "auc_sd": round(float(np.std(aucs)), 3),
                   "mcc_mean": round(float(np.mean(mccs)), 3), "mcc_sd": round(float(np.std(mccs)), 3),
                   "bacc_mean": round(float(np.mean(baccs)), 3), "bacc_sd": round(float(np.std(baccs)), 3)}
            stability.append(row)
            print(f"  {r['arm'].ljust(30)} AUC {row['auc_mean']:.3f} ± {row['auc_sd']:.3f} | "
                  f"MCC {row['mcc_mean']:.3f} ± {row['mcc_sd']:.3f} | "
                  f"BalAcc {row['bacc_mean']:.3f} ± {row['bacc_sd']:.3f}")

    # ---- Best tree arm: feature importance + permutation test ----
    tree_arms = [r for r in results if r["arm"] in fold_store]
    best = max(tree_arms, key=lambda r: (r["metrics"]["roc_auc"], r["metrics"]["mcc"]))
    folds, X_best, kind_best, fs_best = fold_store[best["arm"]]
    names = feature_names(fs_best)
    print(f"\n[VerSync] Best tree arm by pooled OOF ROC-AUC: {best['arm']} "
          f"(AUC {best['metrics']['roc_auc']:.3f}, MCC {best['metrics']['mcc']:.3f})")

    builtin = np.mean([m.feature_importances_ for m, _ in folds], axis=0)
    perm_mean, perm_sd = oof_permutation_importance(folds, X_best, y_sub, oof_table[best["arm"]][sub_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_imp = oof_shap_importance(folds, X_best)

    imp_rows = []
    for j, n in enumerate(names):
        imp_rows.append({
            "feature": n,
            "builtin_importance": round(float(builtin[j]), 5),
            "oof_permutation_auc_drop_mean": round(float(perm_mean[j]), 5),
            "oof_permutation_auc_drop_sd": round(float(perm_sd[j]), 5),
            "oof_mean_abs_shap": round(float(shap_imp[j]), 5) if shap_imp is not None else None,
        })
    imp_rows.sort(key=lambda r: r["oof_permutation_auc_drop_mean"], reverse=True)
    with open(os.path.join(args.results_dir, "feature_importance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(imp_rows[0].keys()))
        w.writeheader()
        w.writerows(imp_rows)
    with open(os.path.join(args.results_dir, "feature_importance.json"), "w") as fh:
        json.dump({"arm": best["arm"], "model": kind_best, "features": fs_best,
                   "selection": "highest pooled OOF ROC-AUC among RF/XGB arms (tie-break MCC)",
                   "importance": imp_rows}, fh, indent=2)
    plot_importance(names, perm_mean, perm_sd, builtin,
                    f"{best['arm']} - top features (n={n_sub}, {int((y_sub == 0).sum())} safe)",
                    os.path.join(args.results_dir, "feature_importance.png"))

    print("\n  Top features (OOF permutation AUC drop | built-in | mean|SHAP|):")
    for r in imp_rows[:10]:
        shap_s = f"{r['oof_mean_abs_shap']:.4f}" if r["oof_mean_abs_shap"] is not None else "n/a"
        print(f"    {r['feature'].ljust(26)} {r['oof_permutation_auc_drop_mean']:+.4f} ± "
              f"{r['oof_permutation_auc_drop_sd']:.4f} | {r['builtin_importance']:.4f} | {shap_s}")

    perm_tests = {}
    if args.perm_iters > 0:
        print(f"\n[VerSync] Label-permutation test ({args.perm_iters} shuffles, full grouped CV each)...")
        lr_name = f"LR SemVer+Graph ({tag})"
        lr_res = next(r for r in results if r["arm"] == lr_name)
        perm_tests[lr_name] = label_permutation_pvalue(
            "LR", feature_matrix(data, "SemVer+Graph", sub_idx), y_sub, g_sub,
            lr_res["metrics"]["roc_auc"], args.perm_iters)
        perm_tests[best["arm"]] = label_permutation_pvalue(
            kind_best, X_best, y_sub, g_sub, best["metrics"]["roc_auc"], args.perm_iters)
        for k, v in perm_tests.items():
            print(f"  {k.ljust(30)} observed AUC {v['observed_auc']:.3f} | null mean {v['null_mean_auc']:.3f} "
                  f"| null 95th pct {v['null_p95_auc']:.3f} | p = {v['p_value']:.3f}")

    # ---- Save results ----
    flat = []
    for r in results:
        row = {"arm": r["arm"], "model": r["model"], "features": r["features"], "n": r["n"], "n_safe": r["n_safe"]}
        for k in METRIC_KEYS:
            row[k] = r["metrics"][k]
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = r["ci95"][k]
        for k in ("tp", "fp", "fn", "tn"):
            row[k] = r["metrics"][k]
        flat.append(row)
    with open(os.path.join(args.results_dir, "model_comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    with open(os.path.join(args.results_dir, "model_comparison.json"), "w") as fh:
        json.dump({
            "protocol": "StratifiedGroupKFold(n_splits=5) grouped by project, pooled OOF probabilities, "
                        "threshold 0.5, 95% bootstrap CIs over pooled OOF predictions",
            "n_total": n_total, "n_graph_subset": n_sub,
            "n_safe_total": int((y == 0).sum()), "n_safe_subset": int((y_sub == 0).sum()),
            "dig_provenance": prov_counts,
            "dependency_found_in_dig": int(data["dep_found"][sub_idx].sum()),
            "bootstrap_iters": args.bootstrap_iters,
            "arms": flat,
        }, fh, indent=2)
    if stability:
        with open(os.path.join(args.results_dir, "seed_stability.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stability[0].keys()))
            w.writeheader()
            w.writerows(stability)
    if perm_tests:
        with open(os.path.join(args.results_dir, "label_permutation_test.json"), "w") as fh:
            json.dump(perm_tests, fh, indent=2)
    with open(os.path.join(args.results_dir, "oof_predictions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "dependency", "label_breaking", "has_dig", "dig_provenance"] + list(oof_table))
        for i, r in enumerate(data["rows"]):
            w.writerow([r["project"], data["deps"][i], int(y[i]), int(data["has_graph"][i]), data["provenance"][i]]
                       + ["" if np.isnan(oof_table[k][i]) else round(float(oof_table[k][i]), 4) for k in oof_table])

    print(f"\n[VerSync] Results written to {os.path.relpath(args.results_dir, REPO_ROOT)}/")
    print("\nNotes:")
    print("  * Every arm uses the same 5-fold StratifiedGroupKFold (grouped by project) with pooled OOF probabilities.")
    print(f"  * Only {int((y_sub == 0).sum())} SAFE samples: safe-class recall moves in steps of "
          f"{1 / max(1, int((y_sub == 0).sum())):.3f}; treat all differences between arms as unreliable "
          "unless CIs separate.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate RF, XGBoost, structural-graph and GNN arms.")
    parser.add_argument("--dataset", default=os.path.join(REPO_ROOT, "person1", "final_verified_dataset.csv"))
    parser.add_argument("--dig-dir", default=os.path.join(REPO_ROOT, "dataset", "digs"))
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=10, help="Model seeds for the stability check (1 = skip)")
    parser.add_argument("--gnn-seeds", type=int, default=5, help="Max seeds for the GNN stability check")
    parser.add_argument("--perm-iters", type=int, default=200, help="Label-permutation test shuffles (0 = skip)")
    parser.add_argument("--no-gnn", action="store_true", help="Skip the GNN arm (no torch needed)")
    args = parser.parse_args()
    args.dataset = os.path.abspath(args.dataset)
    args.dig_dir = os.path.abspath(args.dig_dir) if os.path.isdir(args.dig_dir) else None
    args.results_dir = os.path.abspath(args.results_dir)
    run(args)


if __name__ == "__main__":
    main()
