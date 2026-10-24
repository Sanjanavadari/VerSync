"""
evaluate_localization.py

Paper 2 localization evaluation: does the analyzer's ranked list of impacted files
contain the files that actually failed after a breaking dependency update?

Ground truth : person1/failure_ground_truth.csv  (column `failed_files`, '|'-separated)
Predictions  : DIG reverse traversal from analyzer.py (files containing functions that
               call the updated dependency directly or transitively)
Metrics      : Hit@k, Precision@k, Recall@k for k in {1, 3, 5, 10}, and MRR.

Design decisions (all printed in the report notes):
  * Only BREAKING samples that have code-level ground truth (`failed_files` non-empty) and a
    built DIG are scored. Dependency-level failures (enforcer / lock files, no code location)
    are counted and excluded.
  * A predicted file matches a ground-truth file if their paths are equal or one is a path-segment
    suffix of the other (ground-truth paths come in both 'module/src/main/java/..' and bare
    package form). Bare-basename matching is deliberately not used.
  * Samples whose updated dependency is not found in the DIG get an EMPTY ranking and score 0.
    Results are therefore reported on two populations: all localizable samples, and only those
    where the dependency was matched in the DIG.
  * Precision@k divides by k (standard IR definition), so short rankings are penalised.
  * The main ranking sorts files by (closest call depth, number of direct callers, number of
    affected functions, path); the last key makes it fully deterministic. A second arm
    reproduces the order the shipped CLI would show (probability order, ties in traversal order).
  * Random baselines are expected values over seeded shuffles.

Usage:
    python eval-system/evaluate_localization.py [--bootstrap-iters N] [--shuffles N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzer import _depth_to_probability, extract_graph_features  # noqa: E402

KS = (1, 3, 5, 10)
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Path matching
# ---------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def paths_match(pred_path: str, gt_path: str) -> bool:
    """Equal, or one is a path-segment suffix of the other."""
    a, b = normalize_path(pred_path), normalize_path(gt_path)
    if not a or not b:
        return False
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def _file_of(func_id: str, func_to_file: Dict[str, str]) -> Optional[str]:
    f = func_to_file.get(func_id)
    if f:
        return f
    if func_id.startswith("function:"):
        body = func_id[len("function:"):]
        if "::" in body:
            return body.split("::")[0]
    return None


def rank_files(func_depths: Dict[str, int], func_to_file: Dict[str, str]) -> List[str]:
    """
    Main ranking: files containing impacted functions, ordered by
    (closest call depth asc, #direct callers desc, #affected functions desc, path asc).
    """
    stats: Dict[str, List[int]] = {}  # path -> [min_depth, n_direct, n_funcs]
    for func_id, depth in func_depths.items():
        f = _file_of(func_id, func_to_file)
        if not f:
            continue
        s = stats.setdefault(f, [depth, 0, 0])
        s[0] = min(s[0], depth)
        s[1] += 1 if depth == 1 else 0
        s[2] += 1
    return sorted(stats, key=lambda p: (stats[p][0], -stats[p][1], -stats[p][2], p))


def rank_files_as_shipped(func_depths: Dict[str, int], func_to_file: Dict[str, str]) -> List[str]:
    """
    Order of files implied by the CLI's impacted-files output: functions sorted by impact
    probability (stable, so ties keep traversal order), then de-duplicated by file.
    """
    ordered = sorted(func_depths.items(), key=lambda kv: _depth_to_probability(kv[1]), reverse=True)
    seen: List[str] = []
    for func_id, _ in ordered:
        f = _file_of(func_id, func_to_file)
        if f and f not in seen:
            seen.append(f)
    return seen


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hit_sets(ranked: Sequence[str], gt: Sequence[str]) -> List[FrozenSet[int]]:
    """For each ranked file, the set of ground-truth indices it matches."""
    return [frozenset(i for i, g in enumerate(gt) if paths_match(p, g)) for p in ranked]


def metrics_from_hits(
    hits: Sequence[FrozenSet[int]], n_gt: int, ks: Sequence[int] = KS
) -> Dict[str, float]:
    """Hit@k, Precision@k, Recall@k (k in ks) and reciprocal rank from per-rank hit sets."""
    out: Dict[str, float] = {}
    for k in ks:
        top = hits[:k]
        matched = set().union(*top) if top else set()
        out[f"hit@{k}"] = 1.0 if matched else 0.0
        out[f"precision@{k}"] = sum(1 for h in top if h) / k
        out[f"recall@{k}"] = (len(matched) / n_gt) if n_gt else 0.0
    out["rr"] = 0.0
    for rank, h in enumerate(hits, start=1):
        if h:
            out["rr"] = 1.0 / rank
            break
    return out


def evaluate_ranking(ranked: Sequence[str], gt: Sequence[str], ks: Sequence[int] = KS) -> Dict[str, float]:
    return metrics_from_hits(hit_sets(ranked, gt), len(gt), ks)


def expected_random_metrics(
    hits_in_pool: Sequence[FrozenSet[int]], n_gt: int, shuffles: int, seed: int
) -> Dict[str, float]:
    """Average metrics over seeded random orderings of a pool of files."""
    if not hits_in_pool:
        return metrics_from_hits([], n_gt)
    rng = random.Random(seed)
    order = list(range(len(hits_in_pool)))
    acc: Dict[str, float] = {}
    for _ in range(shuffles):
        rng.shuffle(order)
        m = metrics_from_hits([hits_in_pool[i] for i in order], n_gt)
        for key, val in m.items():
            acc[key] = acc.get(key, 0.0) + val
    return {key: val / shuffles for key, val in acc.items()}


def bootstrap_mean_ci(values: Sequence[float], iters: int, seed: int = 42) -> Tuple[float, float]:
    """95% percentile bootstrap CI for a mean over samples."""
    n = len(values)
    if n < 2 or iters <= 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * iters)], means[min(iters - 1, int(0.975 * iters))]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _raise_csv_field_limit() -> None:
    """
    Ground-truth log excerpts exceed csv's default 128 KB field limit. sys.maxsize overflows
    on Windows (32-bit C long), so back off until the platform accepts a value.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_samples(
    dataset_csv: str, gt_csv: str, dig_report_csv: str, dig_dir: str
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Join BREAKING samples with ground truth and DIGs. Returns (samples, accounting)."""
    _raise_csv_field_limit()

    with open(gt_csv, encoding="utf-8-sig", newline="") as fh:
        gt_rows = {r["breaking_commit"]: r for r in csv.DictReader(fh)}
    with open(dig_report_csv, encoding="utf-8-sig", newline="") as fh:
        report = {r["source_json"]: r for r in csv.DictReader(fh)}
    with open(dataset_csv, encoding="utf-8-sig", newline="") as fh:
        breaking = [r for r in csv.DictReader(fh) if r["classification"].strip().upper() == "BREAKING"]

    acct = {"breaking": len(breaking), "with_gt_row": 0, "with_code_gt": 0, "dependency_level_only": 0,
            "with_built_dig": 0}
    samples: List[Dict[str, Any]] = []

    for row in breaking:
        commit = row["source_json"].replace(".json", "")
        g = gt_rows.get(commit)
        if g is None:
            continue
        acct["with_gt_row"] += 1
        gt_files: List[str] = []
        for f in g["failed_files"].split("|"):
            nf = normalize_path(f)
            if nf and nf not in gt_files:
                gt_files.append(nf)
        if not gt_files:
            acct["dependency_level_only"] += 1
            continue
        acct["with_code_gt"] += 1

        rep = report.get(row["source_json"])
        if not rep or rep.get("status") != "built":
            continue
        dig_path = os.path.join(dig_dir, rep["dig_file"])
        if not os.path.isfile(dig_path):
            continue
        acct["with_built_dig"] += 1
        with open(dig_path, encoding="utf-8") as fh:
            dig = json.load(fh)

        dep = f"{row['dependency_group_id']}:{row['dependency_artifact_id']}"
        feats = extract_graph_features(dig, dep, row["old_version"], row["new_version"])
        universe = sorted({
            (n["id"][len("file:"):] if n["id"].startswith("file:") else n["id"])
            for n in dig.get("nodes", []) if n.get("type") == "FILE"
        })
        samples.append({
            "project": row["project"],
            "commit": commit,
            "dependency": dep,
            "bump": feats["bump_type"],
            "gt_files": gt_files,
            "dep_matched": bool(feats["matched_dep_nodes"]),
            "ranked_main": rank_files(feats["func_depths"], feats["func_to_file"]),
            "ranked_shipped": rank_files_as_shipped(feats["func_depths"], feats["func_to_file"]),
            "universe": universe,
        })
    return samples, acct


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

METRIC_KEYS = [f"{m}@{k}" for m in ("hit", "precision", "recall") for k in KS] + ["rr"]
ARMS = [
    "Depth + fan-in ranking",
    "Analyzer order (as shipped)",
    "Random order, impacted files only",
    "Random order, all DIG files",
]


def score_sample(s: Dict[str, Any], shuffles: int, seed: int) -> Dict[str, Dict[str, float]]:
    gt = s["gt_files"]
    n_gt = len(gt)
    universe_hits = hit_sets(s["universe"], gt)
    return {
        ARMS[0]: evaluate_ranking(s["ranked_main"], gt),
        ARMS[1]: evaluate_ranking(s["ranked_shipped"], gt),
        ARMS[2]: expected_random_metrics(hit_sets(s["ranked_main"], gt), n_gt, shuffles, seed),
        ARMS[3]: expected_random_metrics(universe_hits, n_gt, shuffles, seed),
    }


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def run(args: argparse.Namespace) -> None:
    samples, acct = load_samples(args.dataset, args.ground_truth, args.dig_report, args.dig_dir)

    print("\nSample accounting")
    print(f"  BREAKING samples in dataset ............... {acct['breaking']}")
    print(f"  ... with a ground-truth row ............... {acct['with_gt_row']}")
    print(f"  ... with code-level failed files .......... {acct['with_code_gt']}")
    print(f"  ... dependency-level failure only (excl.) . {acct['dependency_level_only']}")
    print(f"  ... with a built DIG (scored) ............. {acct['with_built_dig']}")

    n_gt_total = sum(len(s["gt_files"]) for s in samples)
    n_gt_in_dig = sum(
        1 for s in samples for g in s["gt_files"] if any(paths_match(u, g) for u in s["universe"])
    )
    n_reach = sum(1 for s in samples if any(any(paths_match(u, g) for u in s["universe"]) for g in s["gt_files"]))
    n_matched = sum(1 for s in samples if s["dep_matched"] and s["ranked_main"])
    print(f"  Ground-truth files present in the DIG ..... {n_gt_in_dig} / {n_gt_total}")
    print(f"  Samples with >=1 ground-truth file in DIG . {n_reach} / {len(samples)}")
    print(f"  Samples where the dependency is matched ... {n_matched} / {len(samples)}")
    n_absent = sum(1 for s in samples if not s["dep_matched"])
    n_no_callers = sum(1 for s in samples if s["dep_matched"] and not s["ranked_main"])
    print(f"    dependency node absent from the DIG ..... {n_absent}")
    print(f"    dependency in DIG but no call sites ..... {n_no_callers}")

    per_sample: List[Dict[str, Any]] = []
    scored: List[Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]] = []
    for i, s in enumerate(samples):
        res = score_sample(s, args.shuffles, args.seed + i)
        scored.append((s, res))
        main = res[ARMS[0]]
        per_sample.append({
            "project": s["project"], "commit": s["commit"], "dependency": s["dependency"],
            "bump": s["bump"], "n_gt_files": len(s["gt_files"]), "n_ranked_files": len(s["ranked_main"]),
            "dep_matched": int(s["dep_matched"]),
            "first_hit_rank": (round(1.0 / main["rr"]) if main["rr"] > 0 else ""),
            **{k: main[k] for k in ("hit@1", "hit@3", "hit@5", "hit@10", "recall@10", "rr")},
        })

    populations = [
        (f"All localizable samples (n={len(scored)})", scored),
        (
            f"Dependency matched in DIG (n={n_matched})",
            [(s, r) for s, r in scored if s["dep_matched"] and s["ranked_main"]],
        ),
    ]

    long_rows: List[Dict[str, Any]] = []
    for pop_name, pop in populations:
        print("\n" + "=" * 118)
        print(pop_name)
        print("=" * 118)
        if not pop:
            print("  (no samples)")
            continue
        means: Dict[str, Dict[str, float]] = {}
        for arm in ARMS:
            means[arm] = {}
            for key in METRIC_KEYS:
                vals = [r[arm][key] for _, r in pop]
                lo, hi = bootstrap_mean_ci(vals, args.bootstrap_iters)
                means[arm][key] = _mean(vals)
                long_rows.append({
                    "population": pop_name, "arm": arm, "metric": key,
                    "value": round(means[arm][key], 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                    "n_samples": len(pop),
                })

        w = 36
        print(f"{'Arm'.ljust(w)} | " + " | ".join(f"Hit@{k}".rjust(6) for k in KS) + " |    MRR")
        print("-" * 118)
        for arm in ARMS:
            m = means[arm]
            print(f"{arm.ljust(w)} | " + " | ".join(f"{m[f'hit@{k}']:.3f}".rjust(6) for k in KS) + f" | {m['rr']:.3f}")
        print()
        print(
            f"{'Arm'.ljust(w)} | "
            + " ".join(f"P@{k}".rjust(6) for k in KS)
            + " | "
            + " ".join(f"R@{k}".rjust(6) for k in KS)
        )
        print("-" * 118)
        for arm in ARMS:
            m = means[arm]
            print(
                f"{arm.ljust(w)} | "
                + " ".join(f"{m[f'precision@{k}']:.3f}".rjust(6) for k in KS)
                + " | "
                + " ".join(f"{m[f'recall@{k}']:.3f}".rjust(6) for k in KS)
            )

    print("\nNotes:")
    print("  * Metrics are macro-averaged over samples; 95% bootstrap CIs are in results/localization_metrics.csv.")
    print("  * Matching is by path equality or path-segment suffix (not basename).")
    print("  * Samples whose dependency is not found in the DIG have an empty ranking and score 0 in the first table.")
    print("  * Precision@k divides by k, so rankings shorter than k are penalised.")
    print("  * Ranking ties are broken deterministically: depth, direct callers, affected functions, path.")
    print("  * Random baselines are expected values over seeded shuffles.")
    print("  * Ground truth comes from test/compile logs; failed test files are scored like any other file.")
    print(f"  * {acct['dependency_level_only']} breaking samples fail at the dependency level (no code location) and are excluded.")

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "localization_metrics.csv")
    sample_path = os.path.join(args.out_dir, "localization_per_sample.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(long_rows)
    with open(sample_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(per_sample[0].keys()))
        wtr.writeheader()
        wtr.writerows(per_sample)
    print(f"\n[VerSync] Saved {metrics_path}")
    print(f"[VerSync] Saved {sample_path}")


def main() -> None:
    root = os.path.join(HERE, "..")
    parser = argparse.ArgumentParser(description="Localization evaluation (Hit@k, Precision@k, Recall@k, MRR).")
    parser.add_argument("--dataset", default=os.path.join(root, "person1", "combined_dataset.csv"))
    parser.add_argument("--ground-truth", default=os.path.join(root, "person1", "failure_ground_truth.csv"))
    parser.add_argument("--dig-dir", default=os.path.join(root, "dataset", "digs"))
    parser.add_argument("--dig-report", default=os.path.join(root, "dataset", "dig_build_report.csv"))
    parser.add_argument("--out-dir", default=os.path.join(root, "results"))
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--shuffles", type=int, default=300, help="Shuffles for the random baselines.")
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
