"""
batch_build_digs.py

Automated DIG Generator for VerSync Dataset Samples.

1. Reads person1/final_verified_dataset.csv and person1/task2_verification_results.json.
2. Prioritizes the scarce SAFE samples first, followed by BREAKING samples.
3. For each sample:
   - Resolves clean git repo URL (e.g. strips GitHub PR suffix '/pull/123' -> '.git').
   - Targets the PRE-UPDATE commit (parent_commit) so we analyze the codebase state
     BEFORE the update/fix was applied.
   - Clones repo via blobless/shallow fetch into a temporary scratch directory.
   - Invokes graph-model/build_dig.py to produce the Dependency-Impact Graph.
   - Saves the resulting DIG to dataset/digs/<project>_<stem>.json.
   - Cleans up the scratch clone to preserve disk space.
4. Idempotent: skips samples whose DIG JSON already exists.

Usage:
    python eval-system/batch_build_digs.py [--limit <int>] [--safe-only]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "graph-model"))

from build_dig import build_dig

GIT_TIMEOUT = 180  # seconds


def clean_repo_url(url: str) -> str:
    """Normalize GitHub PR URLs to standard clone URLs."""
    url = url.strip()
    if "/pull/" in url:
        base = url.split("/pull/")[0]
        return base + ".git" if not base.endswith(".git") else base
    if url.startswith("http://") or url.startswith("https://"):
        return url if url.endswith(".git") else url + ".git"
    return url


def load_dataset_with_commits(
    csv_path: str,
    task2_json_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load samples and attach parent_commit/breaking_commit metadata."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    # Load task2 results for pre-computed parent commits where available
    task2_map = {}
    if task2_json_path and os.path.isfile(task2_json_path):
        try:
            with open(task2_json_path, "r", encoding="utf-8") as fh:
                t2_list = json.load(fh)
                for item in t2_list:
                    if item.get("source_json"):
                        task2_map[item["source_json"]] = item
                    task2_map[(item.get("project"), item.get("dependency_artifact_id"))] = item
        except Exception as exc:
            print(f"[VerSync] Warning: could not parse {task2_json_path}: {exc}", file=sys.stderr)

    enriched = []
    for r in rows:
        src_json = r.get("source_json", "")
        stem = os.path.splitext(src_json)[0] if src_json else ""

        # Derive commit target
        t2 = task2_map.get(src_json) or task2_map.get((r.get("project"), r.get("dependency_artifact_id")))
        parent_commit = t2.get("parent_commit") if t2 else None
        breaking_commit = t2.get("breaking_commit") if t2 else None

        # Fallback commit from 40-char hex source_json stem if valid git SHA
        if not breaking_commit and re.match(r"^[0-9a-fA-F]{40}$", stem):
            breaking_commit = stem

        entry = {
            **r,
            "stem": stem,
            "clean_repo_url": clean_repo_url(r["repo_url"]),
            "parent_commit": parent_commit,
            "breaking_commit": breaking_commit,
        }
        enriched.append(entry)

    # Sort SAFE samples first, then BREAKING
    enriched.sort(key=lambda s: 0 if s.get("classification", "").upper() == "SAFE" else 1)
    return enriched


def clone_and_checkout_commit(
    repo_url: str,
    target_dir: str,
    parent_commit: Optional[str],
    breaking_commit: Optional[str],
) -> bool:
    """Clone repository and checkout the pre-update commit."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)

    # 1. Blobless clone with no checkout for speed & bandwidth efficiency
    clone_cmd = [
        "git", "clone",
        "--filter=blob:none",
        "--no-checkout",
        repo_url,
        target_dir,
    ]
    try:
        res = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=GIT_TIMEOUT)
        if res.returncode != 0:
            print(f"    Git clone failed: {res.stderr[:200]}", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"    Git clone error: {exc}", file=sys.stderr)
        return False

    # 2. Checkout pre-update commit
    # Priority: parent_commit -> breaking_commit~1 -> HEAD
    commit_to_try = parent_commit or (f"{breaking_commit}~1" if breaking_commit else None)

    if commit_to_try:
        # Fetch the specific commit object if blobless didn't include it
        fetch_cmd = ["git", "-C", target_dir, "fetch", "--depth=2", "origin", commit_to_try.split("~")[0]]
        subprocess.run(fetch_cmd, capture_output=True, text=True, timeout=60)

        checkout_cmd = ["git", "-C", target_dir, "checkout", commit_to_try]
        res = subprocess.run(checkout_cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            return True

    # Fallback checkout default branch if specific commit checkout failed
    fallback_cmd = ["git", "-C", target_dir, "checkout", "HEAD"]
    res = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=60)
    return res.returncode == 0


def generate_dig_for_sample(
    sample: Dict[str, Any],
    output_dir: str,
    scratch_dir: str,
) -> Tuple[bool, str]:
    """Generate DIG for one sample and save to output_dir."""
    project = sample["project"]
    stem = sample["stem"]
    out_filename = f"{project}_{stem}.json" if stem else f"{project}.json"
    out_path = os.path.join(output_dir, out_filename)

    if os.path.isfile(out_path):
        return True, f"cached: {out_filename}"

    subfolder_name = f"{project}_{stem}" if stem else project
    repo_dir = os.path.join(scratch_dir, subfolder_name)
    repo_url = sample["clean_repo_url"]
    parent_c = sample.get("parent_commit")
    breaking_c = sample.get("breaking_commit")

    print(f"  Cloning {project} ({repo_url})...", flush=True)
    ok = clone_and_checkout_commit(repo_url, repo_dir, parent_c, breaking_c)
    if not ok:
        shutil.rmtree(repo_dir, ignore_errors=True)
        return False, "clone/checkout failed"

    # Build DIG using graph-model/build_dig.py
    print(f"  Parsing AST & building DIG for {project}...", flush=True)
    try:
        dig, edges, failed = build_dig(repo_dir)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(dig, fh, indent=2)

        n_nodes = len(dig.get("nodes", []))
        n_edges = len(dig.get("edges", []))
        msg = f"built: {n_nodes} nodes, {n_edges} edges -> {out_filename}"
        shutil.rmtree(repo_dir, ignore_errors=True)
        return True, msg
    except Exception as exc:
        shutil.rmtree(repo_dir, ignore_errors=True)
        return False, f"build_dig failed: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate DIGs for VerSync dataset.")
    parser.add_argument(
        "--dataset",
        default=os.path.join(REPO_ROOT, "person1", "final_verified_dataset.csv"),
        help="Path to final_verified_dataset.csv",
    )
    parser.add_argument(
        "--task2-json",
        default=os.path.join(REPO_ROOT, "person1", "task2_verification_results.json"),
        help="Path to task2_verification_results.json",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(REPO_ROOT, "dataset", "digs"),
        help="Output directory for generated DIG JSON files (default: dataset/digs)",
    )
    parser.add_argument(
        "--scratch-dir",
        default=os.path.join(REPO_ROOT, "dataset", "scratch_repos"),
        help="Temporary directory for ephemeral repo checkouts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    parser.add_argument(
        "--safe-only",
        action="store_true",
        help="Only process the 8 SAFE samples",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.scratch_dir, exist_ok=True)

    samples = load_dataset_with_commits(args.dataset, args.task2_json)
    if args.safe_only:
        samples = [s for s in samples if s.get("classification", "").upper() == "SAFE"]
    if args.limit:
        samples = samples[:args.limit]

    n_safe = sum(1 for s in samples if s.get("classification", "").upper() == "SAFE")
    n_breaking = len(samples) - n_safe

    print("=" * 80)
    print(f"[VerSync] Starting batch DIG generation for {len(samples)} samples:")
    print(f"  - Safe samples: {n_safe}")
    print(f"  - Breaking samples: {n_breaking}")
    print(f"  - Output dir: {args.out_dir}")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    for idx, s in enumerate(samples, start=1):
        cls = s.get("classification", "UNKNOWN")
        print(f"\n[{idx}/{len(samples)}] Project: {s['project']} | Update: {s['dependency_artifact_id']} | Type: {cls}")
        ok, detail = generate_dig_for_sample(s, args.out_dir, args.scratch_dir)
        if ok:
            print(f"  [OK] {detail}")
            success_count += 1
        else:
            print(f"  [FAILED] {detail}")
            fail_count += 1

    # Clean up scratch dir
    shutil.rmtree(args.scratch_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print(f"[VerSync] Batch DIG generation completed:")
    print(f"  - Succeeded: {success_count} / {len(samples)}")
    print(f"  - Failed:    {fail_count} / {len(samples)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
