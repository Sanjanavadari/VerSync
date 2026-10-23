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
4. Idempotent: skips samples whose DIG JSON already exists (use --rebuild to force).
5. Writes a per-sample build report (dataset/dig_build_report.csv) recording the
   outcome, the failure stage/reason, which commit was actually analyzed and
   whether that was the intended pre-update commit or a HEAD fallback.

Usage:
    python eval-system/batch_build_digs.py [--limit <int>] [--safe-only]
        [--rebuild] [--only <project> ...] [--no-head-fallback]
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "graph-model"))

from build_dig import build_dig
from extract_call_graph import find_java_files

GIT_TIMEOUT = 180  # seconds
BUILD_TIMEOUT = 900  # seconds per repository for parsing + DIG construction

# Repositories that moved since the dataset was mined. Each entry was verified
# to contain the sample's BUMP breaking commit. Applied here (not in person1/)
# so Person 1's dataset stays untouched.
REPO_URL_OVERRIDES = {
    # 249c3b39 "Bump snakeyaml from 1.17 to 1.31 in /polyglot-yaml"
    "https://github.com/sonatype-nexus-community/polyglot-maven.git": "https://github.com/takari/polyglot-maven.git",
}

REPORT_FIELDS = [
    "project", "source_json", "classification", "status", "stage", "reason",
    "checkout_mode", "checkout_commit", "java_files", "parse_failed_files",
    "nodes", "edges", "seconds", "dig_file",
]


def clean_repo_url(url: str) -> str:
    """Normalize GitHub PR URLs to standard clone URLs."""
    url = url.strip()
    if "/pull/" in url:
        base = url.split("/pull/")[0]
        return base + ".git" if not base.endswith(".git") else base
    if url.startswith("http://") or url.startswith("https://"):
        return url if url.endswith(".git") else url + ".git"
    return url


def pr_number(url: str) -> Optional[str]:
    """Return the GitHub PR number from a '.../pull/<n>' URL, if any."""
    m = re.search(r"/pull/(\d+)", url or "")
    return m.group(1) if m else None


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
            "clean_repo_url": REPO_URL_OVERRIDES.get(clean_repo_url(r["repo_url"]), clean_repo_url(r["repo_url"])),
            "pr_number": pr_number(r["repo_url"]),
            "parent_commit": parent_commit,
            "breaking_commit": breaking_commit,
        }
        enriched.append(entry)

    # Sort SAFE samples first, then BREAKING
    enriched.sort(key=lambda s: 0 if s.get("classification", "").upper() == "SAFE" else 1)
    return enriched


def _git(args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout)


def clone_and_checkout_commit(
    repo_url: str,
    target_dir: str,
    parent_commit: Optional[str],
    breaking_commit: Optional[str],
    pr: Optional[str] = None,
    allow_head_fallback: bool = True,
) -> Tuple[bool, str, str, str]:
    """
    Clone repository and checkout the pre-update commit.

    Returns (ok, checkout_mode, checkout_commit, reason) where checkout_mode is
    'parent_commit', 'breaking~1', 'HEAD_fallback' or '' on failure.
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)

    # 1. Blobless clone with no checkout for speed & bandwidth efficiency
    clone_cmd = [
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        repo_url,
        target_dir,
    ]
    try:
        res = _git(clone_cmd, timeout=GIT_TIMEOUT)
        if res.returncode != 0:
            err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "unknown error"
            print(f"    Git clone failed: {res.stderr[:200]}", file=sys.stderr)
            return False, "", "", f"clone failed: {err[:200]}"
    except subprocess.TimeoutExpired:
        return False, "", "", f"clone timed out after {GIT_TIMEOUT}s"
    except Exception as exc:
        print(f"    Git clone error: {exc}", file=sys.stderr)
        return False, "", "", f"clone error: {exc}"

    # 2. Checkout pre-update commit
    # Priority: parent_commit -> breaking_commit~1 -> HEAD
    candidates = []
    if parent_commit:
        candidates.append(("parent_commit", parent_commit))
    if breaking_commit:
        candidates.append(("breaking~1", f"{breaking_commit}~1"))

    # BUMP breaking commits usually live only on the PR ref (the dependabot/
    # renovate branch is often deleted after merge), so fetch it explicitly.
    if pr:
        try:
            _git(["-C", target_dir, "fetch", "--filter=blob:none", "origin",
                  f"pull/{pr}/head"], timeout=GIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass

    for mode, ref in candidates:
        sha = ref.split("~")[0]
        try:
            # Fetch the specific commit object (+ its parent) if the clone didn't include it
            _git(["-C", target_dir, "fetch", "--filter=blob:none", "--depth=2", "origin", sha])
            res = _git(["-C", target_dir, "checkout", "--quiet", ref], timeout=GIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            continue
        if res.returncode == 0:
            head = _git(["-C", target_dir, "rev-parse", "HEAD"]).stdout.strip()
            return True, mode, head, ""

    tried = ", ".join(r for _, r in candidates) or "none"
    if not allow_head_fallback:
        return False, "", "", f"pre-update commit not reachable (tried: {tried})"

    # Fallback checkout default branch if specific commit checkout failed.
    # This analyzes the *current* code, not the pre-update state, so it is
    # recorded as HEAD_fallback in the report and the DIG metadata.
    res = _git(["-C", target_dir, "checkout", "--quiet", "HEAD"], timeout=GIT_TIMEOUT)
    if res.returncode != 0:
        return False, "", "", f"checkout failed (tried: {tried}, HEAD)"
    head = _git(["-C", target_dir, "rev-parse", "HEAD"]).stdout.strip()
    return True, "HEAD_fallback", head, f"pre-update commit not reachable (tried: {tried})"


def _build_dig_worker(repo_dir: str, queue: "mp.Queue") -> None:
    """Child-process entry point so a hung parse can be killed on timeout."""
    try:
        java_files = len(find_java_files(repo_dir))
        dig, _edges, failed = build_dig(repo_dir)
        queue.put(("ok", dig, java_files, [(os.path.relpath(p, repo_dir), e) for p, e in failed]))
    except BaseException as exc:  # incl. RecursionError on pathological sources
        queue.put(("error", f"{type(exc).__name__}: {exc}", 0, []))


def build_dig_with_timeout(repo_dir: str, timeout: int) -> Tuple[str, Any, int, list]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_build_dig_worker, args=(repo_dir, queue))
    proc.start()
    try:
        result = queue.get(timeout=timeout)
    except Exception:
        proc.kill()
        proc.join()
        return "timeout", f"build_dig exceeded {timeout}s", 0, []
    proc.join()
    return result


def generate_dig_for_sample(
    sample: Dict[str, Any],
    output_dir: str,
    scratch_dir: str,
    rebuild: bool = False,
    allow_head_fallback: bool = True,
) -> Dict[str, Any]:
    """Generate DIG for one sample, save it to output_dir and return a report row."""
    project = sample["project"]
    stem = sample["stem"]
    out_filename = f"{project}_{stem}.json" if stem else f"{project}.json"
    out_path = os.path.join(output_dir, out_filename)
    report = {k: "" for k in REPORT_FIELDS}
    report.update(project=project, source_json=sample.get("source_json", ""),
                  classification=sample.get("classification", ""), dig_file=out_filename)
    t0 = time.time()

    def done(status: str, stage: str = "", reason: str = "") -> Dict[str, Any]:
        report.update(status=status, stage=stage, reason=reason, seconds=round(time.time() - t0, 1))
        return report

    if os.path.isfile(out_path) and not rebuild:
        return done("cached")

    subfolder_name = f"{project}_{stem}" if stem else project
    repo_dir = os.path.join(scratch_dir, subfolder_name)
    repo_url = sample["clean_repo_url"]
    parent_c = sample.get("parent_commit")
    breaking_c = sample.get("breaking_commit")

    print(f"  Cloning {project} ({repo_url})...", flush=True)
    ok, mode, commit, reason = clone_and_checkout_commit(
        repo_url, repo_dir, parent_c, breaking_c, sample.get("pr_number"), allow_head_fallback,
    )
    report.update(checkout_mode=mode, checkout_commit=commit)
    if not ok:
        shutil.rmtree(repo_dir, ignore_errors=True)
        return done("failed", "clone/checkout", reason)
    if mode == "HEAD_fallback":
        print(f"  [WARN] {reason}; analyzing HEAD instead", flush=True)

    # Build DIG using graph-model/build_dig.py
    print(f"  Parsing AST & building DIG for {project}...", flush=True)
    status, payload, n_java, failed = build_dig_with_timeout(repo_dir, BUILD_TIMEOUT)
    shutil.rmtree(repo_dir, ignore_errors=True)
    report.update(java_files=n_java, parse_failed_files=len(failed))
    if status != "ok":
        return done("failed", "build_dig", payload)

    dig = payload
    if n_java == 0:
        return done("failed", "no_java_sources", "no .java files at the checked-out commit")

    dig["build_metadata"] = {
        "checkout_mode": mode,
        "checkout_commit": commit,
        "java_files": n_java,
        "parse_failed_files": len(failed),
        "parse_failed_examples": [f"{p}: {e[:120]}" for p, e in failed[:10]],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dig, fh, indent=2)

    report.update(nodes=len(dig.get("nodes", [])), edges=len(dig.get("edges", [])))
    status = "built_head_fallback" if mode == "HEAD_fallback" else "built"
    return done(status, "", reason)


def write_report(rows: List[Dict[str, Any]], path: str) -> None:
    """Merge rows into the existing report (keyed by dig_file) and rewrite it."""
    existing: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                existing[r["dig_file"]] = r
    for r in rows:
        if r["status"] == "cached" and r["dig_file"] in existing:
            continue  # keep the row describing how the cached DIG was built
        existing[r["dig_file"]] = r
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        w.writeheader()
        for r in sorted(existing.values(), key=lambda x: (x["classification"], x["project"])):
            w.writerow({k: r.get(k, "") for k in REPORT_FIELDS})


def main() -> None:
    global GIT_TIMEOUT, BUILD_TIMEOUT
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
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild DIGs even if the output JSON already exists",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only process these project names",
    )
    parser.add_argument(
        "--no-head-fallback",
        action="store_true",
        help="Fail a sample instead of analyzing HEAD when its pre-update commit is unreachable",
    )
    parser.add_argument(
        "--report",
        default=os.path.join(REPO_ROOT, "dataset", "dig_build_report.csv"),
        help="Per-sample build report CSV (merged across runs)",
    )
    parser.add_argument("--git-timeout", type=int, default=GIT_TIMEOUT, help="Seconds per git clone/fetch")
    parser.add_argument("--build-timeout", type=int, default=BUILD_TIMEOUT, help="Seconds per DIG build")
    args = parser.parse_args()

    GIT_TIMEOUT, BUILD_TIMEOUT = args.git_timeout, args.build_timeout

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.scratch_dir, exist_ok=True)

    samples = load_dataset_with_commits(args.dataset, args.task2_json)
    if args.safe_only:
        samples = [s for s in samples if s.get("classification", "").upper() == "SAFE"]
    if args.only:
        samples = [s for s in samples if s["project"] in set(args.only)]
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
    report_rows = []

    for idx, s in enumerate(samples, start=1):
        cls = s.get("classification", "UNKNOWN")
        print(f"\n[{idx}/{len(samples)}] Project: {s['project']} | Update: {s['dependency_artifact_id']} | Type: {cls}")
        row = generate_dig_for_sample(
            s, args.out_dir, args.scratch_dir,
            rebuild=args.rebuild, allow_head_fallback=not args.no_head_fallback,
        )
        report_rows.append(row)
        if row["status"] == "failed":
            print(f"  [FAILED] {row['stage']}: {row['reason']}")
            fail_count += 1
        else:
            print(f"  [OK] {row['status']}: {row['nodes']} nodes, {row['edges']} edges, "
                  f"{row['parse_failed_files']}/{row['java_files']} files unparsable "
                  f"({row['checkout_mode'] or 'cached'}) -> {row['dig_file']}")
            success_count += 1
        write_report(report_rows, args.report)  # incremental, survives interruption

    # Clean up scratch dir
    shutil.rmtree(args.scratch_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print(f"[VerSync] Batch DIG generation completed:")
    print(f"  - Succeeded: {success_count} / {len(samples)}")
    print(f"  - Failed:    {fail_count} / {len(samples)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
