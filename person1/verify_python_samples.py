"""
Task 5: label the Python generalization dataset with a CI differential.

mine_python_dataset.py collected merged Dependabot/Renovate PRs whose head CI
was green ("GREEN_CI_CANDIDATE") plus one hand-seeded breaking candidate, so the
dataset had no verified breaking sample. This script:

  1. re-checks every existing row and
  2. mines BREAKING samples: Dependabot/Renovate PRs in Python repos whose CI failed.

Label rule. The pre-update state is the PR's merge base and the post-update
state is the PR head; the PR must change only dependency manifests
(requirements*.txt, pyproject.toml, setup.py/cfg, Pipfile, poetry/uv/pdm locks).
  BREAKING: at least one build/test/lint check that succeeded on the pre-update commit
            failed on the post-update commit. Security scans, audits, image pushes and
            deployments do not count. failure_kind says whether a test, lint or other
            build check flipped.
  SAFE:     every check on the post-update commit is green, and the
            pre-update commit is green too.
  UNVERIFIED: anything else (no CI on the pre-update commit, pending checks, a
            failure that was already there before the update, ...).

Like safe_ci_samples.csv, this is CI evidence ("label_source=github_ci_differential"),
not an independent local rebuild. A check can be flaky; failing_checks lists
exactly which checks flipped so a reviewer can look at them.

Usage:
  GITHUB_TOKEN=$(gh auth token) python verify_python_samples.py [--target-breaking 25]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.parse
from collections import Counter
from pathlib import Path

from mine_python_dataset import parse_title, python_dep
from mine_safe_ci_samples import API, BAD_CONCLUSIONS, TRIVIAL_CHECK, Github, latest_checks

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
CSV_OUT = ROOT / "python_generalization_dataset.csv"
JSON_OUT = ROOT / "python_generalization_dataset.json"
FIELDS = ["sample_id", "ecosystem", "project", "repo_url", "pr_url", "pr_number", "title",
          "dependency_name", "old_version", "new_version", "base_sha", "head_sha",
          "ci_status", "ci_checks", "merged_at", "label", "notes",
          "pre_commit_sha", "pre_ci_status", "failing_checks", "failure_kind", "changed_files", "label_source"]
MANIFEST = re.compile(r"(^|/)(requirements[^/]*\.(txt|in)|constraints[^/]*\.txt|pyproject\.toml|"
                      r"setup\.(py|cfg)|Pipfile(\.lock)?|poetry\.lock|uv\.lock|pdm\.lock)$")
# Checks whose failure does not mean the code broke: vulnerability scans, audits,
# image pushes, deployments. A flip in only these is not counted as BREAKING.
NON_FUNCTIONAL = re.compile(r"(?i)(security|audit|codeql|scan|snyk|dependency-review|deploy|publish|"
                            r"push|release|docker|container|image|pages|website|docs?\b)")
LINT = re.compile(r"(?i)(lint|ruff|flake8|pylint|mypy|pyright|black|isort|pre-commit|hooks|format|style|typecheck)")
TEST = re.compile(r"(?i)(test|pytest|unit|tox|suite|validate|simulation|contracts)")

SEARCHES = [
    "is:pr author:app/dependabot language:Python status:failure is:closed",
    "is:pr author:app/renovate language:Python status:failure is:closed",
    "is:pr author:app/dependabot language:Python status:failure is:open",
]


def summarize(checks: dict[str, tuple[str, str]]) -> str:
    if not checks:
        return "NO_CI"
    if any(st != "completed" for st, _ in checks.values()):
        return "PENDING"
    if any(c in BAD_CONCLUSIONS or c == "" for _, c in checks.values()):
        return "NOT_GREEN"
    if not any(c == "success" and not TRIVIAL_CHECK.search(n) for n, (_, c) in checks.items()):
        return "NO_BUILD_CHECK"
    return "GREEN"


def evaluate(gh: Github, pr_url: str) -> dict:
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/pull/(\d+)", pr_url)
    if not m:
        return {"label": "UNVERIFIED", "notes": "not a PR URL"}
    repo, number = m.group(1), m.group(2)
    pr = gh.get(f"{API}/repos/{repo}/pulls/{number}")
    if not pr:
        return {"label": "UNVERIFIED", "notes": "PR not found"}
    dep, old, new = parse_title(pr.get("title", ""))
    files = gh.get(f"{API}/repos/{repo}/pulls/{number}/files?per_page=100") or []
    names = [f.get("filename", "") for f in files]
    head = pr["head"]["sha"]
    cmp = gh.get(f"{API}/repos/{repo}/compare/{pr['base']['sha']}...{head}") or {}
    pre = (cmp.get("merge_base_commit") or {}).get("sha", "")
    post_checks = latest_checks(gh, repo, head)
    pre_checks = latest_checks(gh, repo, pre) if pre else {}
    post_status, pre_status = summarize(post_checks), summarize(pre_checks)
    flipped = sorted(n for n, (st, c) in post_checks.items()
                     if st == "completed" and c in BAD_CONCLUSIONS and not TRIVIAL_CHECK.search(n)
                     and not NON_FUNCTIONAL.search(n) and pre_checks.get(n, ("", ""))[1] == "success")
    kind = ("test" if any(TEST.search(n) for n in flipped) else
            "lint" if flipped and all(LINT.search(n) for n in flipped) else
            "build" if flipped else "")

    row = {
        "project": repo.split("/")[1], "repo_url": f"https://github.com/{repo}", "pr_url": pr["html_url"],
        "pr_number": pr["number"], "title": pr.get("title", ""), "dependency_name": dep,
        "old_version": old, "new_version": new, "base_sha": pr["base"]["sha"], "head_sha": head,
        "ci_status": post_status, "ci_checks": len(post_checks), "merged_at": pr.get("merged_at") or "",
        "pre_commit_sha": pre, "pre_ci_status": pre_status, "failing_checks": "|".join(flipped),
        "failure_kind": kind,
        "changed_files": "|".join(names), "label_source": "github_ci_differential",
    }
    if not dep or not python_dep(dep):
        return {**row, "label": "UNVERIFIED", "notes": "title is not a single Python dependency update"}
    if not names or any(not MANIFEST.search(n) for n in names):
        return {**row, "label": "UNVERIFIED", "notes": "changes files other than dependency manifests"}
    if post_status == "PENDING":
        return {**row, "label": "UNVERIFIED", "notes": "post-update checks still pending"}
    if flipped:
        return {**row, "label": "BREAKING",
                "notes": f"{len(flipped)} check(s) green before the update and failing after it"}
    if post_status == "GREEN" and pre_status == "GREEN":
        return {**row, "label": "SAFE", "notes": "all checks green before and after the update"}
    return {**row, "label": "UNVERIFIED",
            "notes": f"no clean differential (pre={pre_status}, post={post_status})"}


def main() -> None:
    ap = argparse.ArgumentParser(description="CI-differential labels for the Python dataset")
    ap.add_argument("--target-breaking", type=int, default=25)
    ap.add_argument("--max-inspect", type=int, default=250)
    args = ap.parse_args()

    gh = Github()
    existing = list(csv.DictReader(CSV_OUT.open(encoding="utf-8-sig"))) if CSV_OUT.exists() else []
    rows: dict[str, dict] = {}
    for r in existing:
        res = evaluate(gh, r["pr_url"])
        if r.get("label") == "BREAKING_CANDIDATE" and res["label"] != "BREAKING":
            res["notes"] = "hand-seeded candidate, not confirmed by CI: " + res.get("notes", "")
        rows[r["pr_url"]] = {**r, **res}
        print(f"existing {r['pr_url']}: {res['label']} ({res.get('notes', '')})")

    n_breaking = sum(r["label"] == "BREAKING" for r in rows.values())
    inspected = 0
    for q in SEARCHES:
        for page in (1, 2, 3):
            if n_breaking >= args.target_breaking or inspected >= args.max_inspect:
                break
            url = f"{API}/search/issues?" + urllib.parse.urlencode(
                {"q": q, "per_page": 100, "page": page, "sort": "created", "order": "desc"})
            items = (gh.get(url, search=True) or {}).get("items", [])
            for item in items:
                if n_breaking >= args.target_breaking or inspected >= args.max_inspect:
                    break
                dep, _, _ = parse_title(item.get("title", ""))
                pr_url = item.get("html_url", "")
                if not python_dep(dep) or pr_url in rows:
                    continue
                repo = pr_url.split("/pull/")[0]
                if sum(r["repo_url"] == repo and r["label"] == "BREAKING" for r in rows.values()) >= 2:
                    continue  # at most 2 breaking samples per repository
                inspected += 1
                try:
                    res = evaluate(gh, pr_url)
                except urllib.error.HTTPError as exc:
                    print(f"  {pr_url}: HTTP {exc.code}")
                    continue
                if res["label"] == "BREAKING":
                    rows[pr_url] = {"ecosystem": "python", **res}
                    n_breaking += 1
                    print(f"  BREAKING {n_breaking}: {res['project']} {res['dependency_name']} "
                          f"{res['old_version']}->{res['new_version']} [{res['failing_checks']}]")
            if len(items) < 100:
                break
    gh.save()

    out = list(rows.values())
    for i, r in enumerate(out, 1):
        r["sample_id"] = i
        r.setdefault("ecosystem", "python")
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    JSON_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nInspected {inspected} new PRs")
    print("Labels:", dict(Counter(r["label"] for r in out)))
    print("Wrote", CSV_OUT.name, JSON_OUT.name)


if __name__ == "__main__":
    main()
