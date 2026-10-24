"""
calibrate_triage.py

Checks the SAFE / RISKY / CRITICAL triage tiers against the labelled samples that have a DIG.

It runs the same functions the CLI uses (extract_graph_features, _compute_graph_heuristic_score,
compute_triage), so this table cannot drift from the shipped behaviour. It reports:

  A. Score-only tiers (thresholds SAFE_THRESHOLD / RISKY_THRESHOLD), the pre-Task-5 behaviour.
  B. Triage tiers (score + graph-evidence rules), the Task 5 behaviour.
  C. Evidence level vs. class.
  D. Sensitivity of the RISKY/CRITICAL boundary (the only boundary with enough data to check).

Important limits (also printed): only samples with a DIG are scored, SAFE samples are scarce, and
the original SAFE samples all come from projects that also contributed BREAKING samples. The tier
boundaries are therefore a sanity check, not a statistically calibrated cut-off.

Usage:
    python eval-system/calibrate_triage.py [--dataset <csv>] [--out results/triage_calibration.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzer import (  # noqa: E402
    RISKY_THRESHOLD,
    SAFE_THRESHOLD,
    _categorize_risk,
    _compute_graph_heuristic_score,
    compute_triage,
    extract_graph_features,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TIERS = ["SAFE", "RISKY", "CRITICAL"]


def load_scored_samples(dataset_csv: str, dig_report_csv: str, dig_dir: str) -> List[Dict[str, Any]]:
    limit = 2**31 - 1
    csv.field_size_limit(limit)
    with open(dig_report_csv, encoding="utf-8-sig", newline="") as fh:
        report = {r["source_json"]: r for r in csv.DictReader(fh)}
    samples: List[Dict[str, Any]] = []
    with open(dataset_csv, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rep = report.get(row.get("source_json", "")) if row.get("source_json") else None
            if not rep or rep.get("status") != "built":
                continue
            path = os.path.join(dig_dir, rep["dig_file"])
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as dh:
                dig = json.load(dh)
            dep = f"{row['dependency_group_id']}:{row['dependency_artifact_id']}"
            feats = extract_graph_features(dig, dep, row["old_version"], row["new_version"])
            score = _compute_graph_heuristic_score(feats)
            triage = compute_triage(feats, score, dep, row["old_version"], row["new_version"])
            samples.append({
                "project": row["project"],
                "dependency": dep,
                "breaking": 1 if row["classification"].strip().upper() == "BREAKING" else 0,
                "score": score,
                "score_tier": _categorize_risk(score),
                "triage_tier": triage["tier"],
                "evidence": triage["evidence"],
            })
    return samples


def _counts(samples: List[Dict[str, Any]], key: str, values: List[str]) -> List[Dict[str, Any]]:
    out = []
    for v in values:
        sub = [s for s in samples if s[key] == v]
        brk = sum(s["breaking"] for s in sub)
        out.append({
            "group": v,
            "n": len(sub),
            "n_breaking": brk,
            "n_safe": len(sub) - brk,
            "breaking_rate": (brk / len(sub)) if sub else None,
        })
    return out


def _print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    print(f"  {'group':<26} {'n':>4} {'breaking':>9} {'safe':>5} {'breaking rate':>14}")
    for r in rows:
        rate = "n/a" if r["breaking_rate"] is None else f"{r['breaking_rate']:.0%}"
        print(f"  {r['group']:<26} {r['n']:>4} {r['n_breaking']:>9} {r['n_safe']:>5} {rate:>14}")


def main() -> None:
    root = os.path.join(HERE, "..")
    ap = argparse.ArgumentParser(description="Check triage tiers against labelled DIG-backed samples.")
    ap.add_argument("--dataset", default=os.path.join(root, "person1", "combined_dataset.csv"))
    ap.add_argument("--dig-report", default=os.path.join(root, "dataset", "dig_build_report.csv"))
    ap.add_argument("--dig-dir", default=os.path.join(root, "dataset", "digs"))
    ap.add_argument("--out", default=os.path.join(root, "results", "triage_calibration.csv"))
    args = ap.parse_args()

    samples = load_scored_samples(args.dataset, args.dig_report, args.dig_dir)
    n_brk = sum(s["breaking"] for s in samples)
    print(f"Samples with a DIG: {len(samples)} ({n_brk} BREAKING, {len(samples) - n_brk} SAFE)")
    print(f"Score thresholds: SAFE < {SAFE_THRESHOLD} <= RISKY < {RISKY_THRESHOLD} <= CRITICAL")

    table_a = _counts(samples, "score_tier", TIERS)
    table_b = _counts(samples, "triage_tier", TIERS)
    table_c = _counts(samples, "evidence", ["call-sites-found", "no-call-sites", "dependency-not-in-graph"])
    _print_table("A. Score-only tiers (behaviour before Task 5)", table_a)
    _print_table("B. Triage tiers (score + graph-evidence rules, Task 5)", table_b)
    _print_table("C. Evidence level", table_c)

    print("\nD. Sensitivity of the CRITICAL boundary (score >= t)")
    print(f"  {'t':>5} {'n above':>8} {'breaking':>9} {'safe':>5}")
    sweep = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        above = [s for s in samples if s["score"] >= t]
        brk = sum(s["breaking"] for s in above)
        sweep.append({"group": f"critical_threshold_{t}", "n": len(above), "n_breaking": brk,
                      "n_safe": len(above) - brk, "breaking_rate": (brk / len(above)) if above else None})
        print(f"  {t:>5} {len(above):>8} {brk:>9} {len(above) - brk:>5}")

    safe_a = next(r for r in table_a if r["group"] == "SAFE")
    safe_b = next(r for r in table_b if r["group"] == "SAFE")
    print("\nReading the tables")
    print(f"  * Score-only SAFE tier: {safe_a['n_breaking']} of {safe_a['n']} samples were actually breaking.")
    print(f"  * Triage SAFE tier:     {safe_b['n_breaking']} of {safe_b['n']} samples were actually breaking.")
    print("  * The SAFE tier cannot be validated here: there are no SAFE samples with call-site evidence of safety,")
    print("    and the class balance of this subset is far from the real-world mix.")
    print("  * 'dependency-not-in-graph' is mostly breaking in this data, so it is never reported as SAFE.")
    print("  * Treat boundaries as sanity-checked, not statistically calibrated (few SAFE samples).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["table", "group", "n", "n_breaking", "n_safe", "breaking_rate"])
        for name, rows in (("score_only_tier", table_a), ("triage_tier", table_b),
                           ("evidence", table_c), ("critical_sweep", sweep)):
            for r in rows:
                rate = "" if r["breaking_rate"] is None else round(r["breaking_rate"], 4)
                w.writerow([name, r["group"], r["n"], r["n_breaking"], r["n_safe"], rate])
    print(f"\n[VerSync] Saved {args.out}")


if __name__ == "__main__":
    main()
