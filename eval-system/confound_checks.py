#!/usr/bin/env python3
"""
confound_checks.py

Checks whether the graph features' gain on the combined dataset is real signal or
an artifact of how the 45 CI-labelled SAFE samples were collected.

  1. Within-project pairwise AUC: for every project with both a BREAKING and a
     SAFE sample, the share of (breaking, safe) pairs in which the breaking one
     gets the higher out-of-fold P(breaking). Repo identity cannot help here.
     Reads results/oof_predictions.csv (written by train_models.py).
  2. Updated-artifact mix by class and label source (library, Maven plugin,
     BOM/parent, test library). Maven plugins never appear in Java code, so
     a class imbalance in them inflates "dependency not in graph" features.
  3. Writes the combined dataset without Maven-plugin updates, and re-runs the
     train_baseline.py ablation on it and on the docker-labelled samples only.

Usage:
    python eval-system/confound_checks.py
"""

import argparse
import collections
import csv
import itertools
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# Arm-name prefixes in oof_predictions.csv; the "(n=...)" suffix depends on the dataset.
ARMS = ["LR SemVer (n=", "LR Graph (n=", "LR SemVer+Graph (n=", "RF SemVer+Graph (n=",
        "XGB SemVer+Graph (n=", "GNN impact-graph+SemVer"]


def artifact_kind(group_id: str, artifact_id: str) -> str:
    if "plugin" in artifact_id and ("maven" in artifact_id or "maven" in group_id):
        return "maven-plugin"
    if re.search(r"bom|dependencies$|parent", artifact_id):
        return "bom/parent"
    if re.search(r"junit|mockito|assertj|test|hamcrest", artifact_id + group_id):
        return "test-lib"
    return "library"


def within_project_auc(oof_rows, arm):
    by_project = collections.defaultdict(list)
    for r in oof_rows:
        if r.get("has_dig") == "1" and r.get(arm, "") != "":
            by_project[r["project"]].append((int(r["label_breaking"]), float(r[arm])))
    wins = ties = pairs = 0
    for samples in by_project.values():
        for (_, pb), (_, ps) in itertools.product([s for s in samples if s[0] == 1],
                                                  [s for s in samples if s[0] == 0]):
            pairs += 1
            wins += pb > ps
            ties += pb == ps
    projects = sum(1 for s in by_project.values() if len({l for l, _ in s}) == 2)
    return (wins + 0.5 * ties) / pairs if pairs else float("nan"), pairs, projects


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=os.path.join(ROOT, "person1", "combined_dataset.csv"))
    ap.add_argument("--oof", default=os.path.join(ROOT, "results", "oof_predictions.csv"))
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--skip-ablations", action="store_true")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(args.oof, encoding="utf-8-sig", newline="") as fh:
        oof = list(csv.DictReader(fh))

    out = []
    print("1. Within-project pairwise AUC (breaking vs safe of the same project, OOF probabilities)")
    columns = list(oof[0]) if oof else []
    for prefix in ARMS:
        # The graph-subset arm, not an "(all N)" arm, so every arm is scored on the same samples.
        arm = next((c for c in columns if c.startswith(prefix) and "all" not in c), None)
        if arm:
            auc, pairs, projects = within_project_auc(oof, arm)
            print(f"   {arm:28s} AUC {auc:.3f}  ({pairs} pairs, {projects} projects)")
            out.append({"check": "within_project_auc", "group": arm, "value": round(auc, 3), "n": pairs})

    print("\n2. Updated-artifact kind by class / label source")
    kinds = collections.Counter((r["classification"], r.get("label_source", ""),
                                 artifact_kind(r["dependency_group_id"], r["dependency_artifact_id"])) for r in rows)
    for (cls, src, kind), n in sorted(kinds.items()):
        print(f"   {cls:8s} {src:26s} {kind:12s} {n}")
        out.append({"check": "artifact_kind", "group": f"{cls}/{src}/{kind}", "value": n, "n": ""})

    path = os.path.join(args.results_dir, "confound_checks.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "group", "value", "n"], lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"\n[VerSync] Saved {path}")

    if args.skip_ablations:
        return
    no_plugin = os.path.join(args.results_dir, "_combined_no_maven_plugins.csv")
    kept = [r for r in rows if artifact_kind(r["dependency_group_id"], r["dependency_artifact_id"]) != "maven-plugin"]
    with open(no_plugin, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(kept)
    runs = [
        ("no Maven-plugin updates", no_plugin, "ablation_no_maven_plugins.csv"),
        ("docker labels only", os.path.join(ROOT, "person1", "final_verified_dataset.csv"), "ablation_docker_labels.csv"),
    ]
    try:
        for label, dataset, name in runs:
            print(f"\n3. Ablation, {label}")
            subprocess.run([sys.executable, os.path.join(HERE, "train_baseline.py"), "--dataset", dataset,
                            "--out", os.path.join(args.results_dir, name),
                            "--strata-out", os.path.join(args.results_dir, "_strata_tmp.csv")], check=True)
    finally:
        for tmp in (no_plugin, os.path.join(args.results_dir, "_strata_tmp.csv")):
            if os.path.exists(tmp):
                os.remove(tmp)


if __name__ == "__main__":
    main()
