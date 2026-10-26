"""
Step 5: mine SAFE updates that match the BREAKING samples in kind and bump type.

The 45 SAFE samples from mine_safe_ci_samples.py come from the right repositories,
but their updates differ from the BREAKING ones (results/README.md, confound checks):
  * bump type: SAFE has 2 MAJOR bumps, BREAKING has 23;
  * artifact kind: 12 of the 45 update a Maven plugin, vs 1 of 51 BREAKING. Plugins
    never appear in Java code, so "dependency not in the graph" partly means "plugin".

This script mines more SAFE samples from the same repositories, keeping only MAJOR
bumps of libraries (no Maven plugins, BOMs or parent POMs: the artifact's POM must have
jar/bundle packaging), newest PRs first. Older PRs would match the BUMP dates better, but a
first run that went oldest-first found no CI record for 160 of them (GitHub does not keep
check results that long), so they cannot be verified. Every accepted PR passes exactly the same checks
as Task 1 (inspect_pr in mine_safe_ci_samples.py): merged bot PR, one Maven coordinate,
only pom.xml changed, green CI on the head with a real build check, pre-update commit
(merge base) not red.

Usage:
  GITHUB_TOKEN=$(gh auth token) python mine_matched_safe_samples.py [--per-repo 2] [--max-pages 20]

Outputs:
  matched_safe_samples.csv / .json    accepted SAFE samples (mining_round = matched_major)
  matched_safe_rejections.csv         every inspected PR and why it was rejected
  combined_dataset_matched.csv        combined_dataset.csv + matched_safe_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import sys
import urllib.error
from collections import Counter
from pathlib import Path

from api_diff_extractor import read_pom
from mine_safe_ci_samples import (FIELDS, HEADERS, SKIP_REPOS, Github, canonical_repo, inspect_pr,
                                  list_bot_prs, parse_update, repo_of)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "eval-system"))
from analyzer import classify_semver_bump  # noqa: E402  (same bump rule the models use)
from confound_checks import artifact_kind  # noqa: E402  (same kind rule the confound checks use)

COMBINED = ROOT / "combined_dataset.csv"
OUT_CSV = ROOT / "matched_safe_samples.csv"
OUT_JSON = ROOT / "matched_safe_samples.json"
REJECT_CSV = ROOT / "matched_safe_rejections.csv"
MATCHED_CSV = ROOT / "combined_dataset_matched.csv"
OUT_FIELDS = FIELDS + ["mining_round"]


def wanted(item: dict) -> tuple[bool, str]:
    """Cheap filter on the PR title/body before any extra API call."""
    dep, old, new = parse_update(item.get("title", ""), item.get("body") or "")
    if not dep or not old or not new:
        return False, "title is not a single Maven coordinate update"
    group, _, artifact = dep.rpartition(":")
    if artifact_kind(group, artifact) != "library":
        return False, f"not a library update ({artifact_kind(group, artifact)})"
    bump = classify_semver_bump(old, new)
    if bump != "MAJOR":
        return False, f"{bump} bump"
    return True, ""


def is_library_jar(group: str, artifact: str, version: str) -> tuple[bool, str]:
    """True if the new version's POM declares jar-like packaging (not pom / maven-plugin / hpi)."""
    try:
        packaging = read_pom(group, artifact, version)["packaging"]
    except Exception as exc:  # POM not on the configured repositories
        return False, f"POM not readable ({type(exc).__name__})"
    if packaging in ("jar", "bundle"):
        return True, packaging
    return False, f"packaging {packaging}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-repo", type=int, default=2)
    ap.add_argument("--max-pages", type=int, default=20, help="Pages of 100 closed PRs to scan per repo")
    ap.add_argument("--max-inspect-per-repo", type=int, default=12)
    args = ap.parse_args()
    if "Authorization" not in HEADERS:
        print("Set GITHUB_TOKEN (e.g. GITHUB_TOKEN=$(gh auth token)); the anonymous rate limit is too low.")
        sys.exit(1)

    rows = list(csv.DictReader(COMBINED.open(encoding="utf-8-sig")))
    known = {(r.get("github_repo") or repo_of(r["repo_url"]), r["dependency_group_id"], r["dependency_artifact_id"],
              r["old_version"], r["new_version"]) for r in rows}
    known_artifacts = {(r.get("github_repo") or repo_of(r["repo_url"]), r["dependency_artifact_id"]) for r in rows}
    repos = sorted({repo_of(r["repo_url"]) for r in rows if r["classification"] == "BREAKING"} - SKIP_REPOS - {""})
    print(f"Mining {len(repos)} repositories that have BREAKING samples (MAJOR library bumps only)")

    gh = Github()
    selected: list[dict] = []
    rejections = []
    for i, repo in enumerate(repos, 1):
        canon = canonical_repo(gh, repo)
        if not canon:
            print(f"[{i}/{len(repos)}] {repo}: repository not found, skipped")
            continue
        repo = canon
        items = list_bot_prs(gh, repo, max_pages=args.max_pages)
        candidates = []
        for item in items:
            ok, reason = wanted(item)
            if ok:
                candidates.append(item)
        candidates.sort(key=lambda it: it.get("merged_at") or "", reverse=True)  # newest first: CI still recorded
        accepted: list[dict] = []
        for item in candidates[:args.max_inspect_per_repo]:
            if len(accepted) >= args.per_repo:
                break
            try:
                row, reason = inspect_pr(gh, repo, item, known)
            except urllib.error.HTTPError as exc:
                row, reason = None, f"HTTP {exc.code}"
            except (OSError, http.client.HTTPException) as exc:
                row, reason = None, f"network error: {exc}"
            if row:
                ok, packaging = is_library_jar(row["dependency_group_id"], row["dependency_artifact_id"],
                                               row["new_version"])
                if not ok:
                    row, reason = None, f"not a library jar ({packaging})"
            if row and ((repo, row["dependency_artifact_id"]) in known_artifacts
                        or any(r["dependency_artifact_id"] == row["dependency_artifact_id"] for r in accepted)):
                row, reason = None, "same artifact already sampled for this repo"
            if row:
                accepted.append({**row, "mining_round": "matched_major"})
            rejections.append({"github_repo": repo, "pr_url": item.get("html_url", ""),
                               "title": item.get("title", ""), "result": reason})
        selected.extend(accepted)
        print(f"[{i}/{len(repos)}] {repo}: {len(items)} bot PRs, {len(candidates)} MAJOR library bumps, "
              f"{len(accepted)} accepted")
        gh.save()
    gh.save()

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(selected)
    OUT_JSON.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    with REJECT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["github_repo", "pr_url", "title", "result"], lineterminator="\n")
        w.writeheader()
        w.writerows(rejections)
    with MATCHED_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r.get(k, "") for k in FIELDS},
                        "mining_round": "task1_ci" if r.get("label_source") == "github_ci" else "bump"})
        w.writerows(selected)

    print(f"\nSelected {len(selected)} SAFE samples from {len({r['github_repo'] for r in selected})} repos")
    print("Merged year:", dict(sorted(Counter(r["merged_at"][:4] for r in selected).items())))
    print("Pre-update CI:", dict(Counter(r["pre_ci_status"] for r in selected)))
    print("Rejections:", Counter(r["result"].split(":")[0] for r in rejections).most_common(8))
    print(f"Wrote {OUT_CSV.name}, {REJECT_CSV.name}, {MATCHED_CSV.name} ({len(rows) + len(selected)} rows)")


if __name__ == "__main__":
    main()
