"""
Task 1: mine SAFE dependency updates = merged Dependabot/Renovate PRs that passed CI.

Why these repos: the 8 SAFE samples in final_verified_dataset.csv come from
projects that have no BREAKING sample, so any repo-level property separates
the classes (see results/README.md on Person2/ML-models). Mining SAFE updates
from the *same* repositories as the BREAKING samples removes that confound.

A PR is accepted as SAFE only if all of these hold:
  * author is dependabot[bot] or renovate[bot], PR is merged
  * the title names exactly one Maven coordinate (groupId:artifactId) with
    old and new versions (grouped updates like "bump the maven group" are skipped)
  * every changed file is a pom.xml (no code changes mixed in)
  * CI on the PR head commit is GREEN: every check run / commit status
    (latest per name) finished as success/neutral/skipped, none failed, and at
    least one successful check looks like a real build (not only CLA, codecov,
    labelers, ...)
The CI state of the pre-update commit (the PR's merge base) is recorded
too but not required: many repos only run CI on PRs.

Labels are therefore "CI-verified SAFE", weaker than BUMP's docker
reproduction; every row says so in label_source.

Usage:
  GITHUB_TOKEN=$(gh auth token) python mine_safe_ci_samples.py [--target 45] [--per-repo 2]

Outputs:
  safe_ci_samples.csv / .json   accepted SAFE samples
  safe_ci_rejections.csv        every inspected PR and why it was rejected
  combined_dataset.csv          final_verified_dataset.csv + safe_ci_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "final_verified_dataset.csv"
OUT_CSV = ROOT / "safe_ci_samples.csv"
OUT_JSON = ROOT / "safe_ci_samples.json"
REJECT_CSV = ROOT / "safe_ci_rejections.csv"
COMBINED_CSV = ROOT / "combined_dataset.csv"
CACHE_FILE = ROOT / ".cache" / "github_api_cache.json"
API = "https://api.github.com"

HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "VerSync-Safe-Miner",
    "X-GitHub-Api-Version": "2022-11-28",
}
if os.getenv("GITHUB_TOKEN"):
    HEADERS["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]

# Repos that moved since BUMP was mined, or rows whose repo_url is not a PR URL.
REPO_OVERRIDES = {
    "sonatype-nexus-community/polyglot-maven": "takari/polyglot-maven",
}
# Repos we cannot use: deleted, or too large to build a DIG for.
SKIP_REPOS = {"artipie/docker-adapter", "googleapis/google-cloud-java"}

TITLE_PATTERNS = [
    # Dependabot: "Bump g:a from 1.0 to 1.1", "build(deps): bump g:a from ... to ... in /x"
    # Some repos configure Dependabot to drop the groupId ("bump spring-core from ..."); the
    # groupId is then recovered from the pom.xml patch or Maven Central (resolve_group).
    re.compile(r"(?i)\bbump\s+(?P<dep>[\w.\-]+(?::[\w.\-]+)?)\s+from\s+(?P<old>\S+)\s+to\s+(?P<new>\S+)"),
    # Renovate with explicit versions
    re.compile(r"(?i)\bupdate\s+(?:dependency|plugin)\s+(?P<dep>[\w.\-]+:[\w.\-]+)\s+from\s+v?(?P<old>\S+)\s+to\s+v?(?P<new>\S+)"),
]
RENOVATE_TITLE = re.compile(r"(?i)\bupdate\s+(?:dependency|plugin)\s+(?P<dep>[\w.\-]+:[\w.\-]+)\s+to\s+v?(?P<new>[\w.\-]+)")
# Renovate body table cell: `1.2.3` -> `1.2.4`
RENOVATE_BODY = re.compile(r"`v?(?P<old>[\w.\-]+)`\s*(?:->|→|&rarr;)\s*`v?(?P<new>[\w.\-]+)`")

GOOD_CONCLUSIONS = {"success", "neutral", "skipped"}
BAD_CONCLUSIONS = {"failure", "cancelled", "timed_out", "action_required", "startup_failure", "stale"}
# Checks that pass without building the code.
TRIVIAL_CHECK = re.compile(
    r"(?i)(\bcla\b|cla/|codecov|coveralls|license/|dependabot|renovate|mergify|\bwip\b|semantic|"
    r"labeler|label|stale|dco|title|auto-?merge|changelog|release-drafter|gitguardian|snyk|"
    r"lgtm|welcome|greeting|triage|assign|netlify|vercel|sonarcloud code analysis|pr-lint)"
)

FIELDS = [
    "project", "repo_url", "dependency_group_id", "dependency_artifact_id", "old_version",
    "new_version", "java_version", "failure_category", "pre_update_status", "post_update_status",
    "classification", "source_json",
    "label_source", "github_repo", "pr_number", "bot", "pre_commit_sha", "post_commit_sha",
    "merge_commit_sha", "merged_at", "changed_files", "ci_checks", "pre_ci_status",
]


class Github:
    def __init__(self) -> None:
        try:
            self.cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.cache = {}
        self.dirty = 0

    def get(self, url: str, search: bool = False):
        if url in self.cache:
            return self.cache[url]
        for attempt in range(4):
            req = urllib.request.Request(url, headers=HEADERS)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 429) and attempt < 3:
                    reset = exc.headers.get("X-RateLimit-Reset")
                    remaining = exc.headers.get("X-RateLimit-Remaining")
                    if reset and remaining == "0":
                        wait = max(5, int(reset) - int(time.time()) + 2)
                    else:  # secondary rate limit: back off for a minute or more
                        wait = int(exc.headers.get("Retry-After") or 60 * (attempt + 1))
                    print(f"    rate limited, sleeping {min(wait, 900)}s")
                    time.sleep(min(wait, 900))
                    continue
                if exc.code in (404, 410, 422):
                    data = None
                    break
                raise
            except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as exc:
                if attempt == 3:
                    raise
                print(f"    network error ({exc}), retrying")
                time.sleep(10 * (attempt + 1))
        self.cache[url] = data
        self.dirty += 1
        if self.dirty % 200 == 0:  # the cache is large; rewriting it often is slow
            self.save()
        time.sleep(2.1 if search else 0.25)  # stay clear of secondary rate limits
        return data

    def save(self) -> None:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(self.cache), encoding="utf-8")


def repo_of(url: str) -> str:
    m = re.match(r"https://github\.com/([^/]+/[^/.]+)", url.strip())
    if not m:
        return ""
    return REPO_OVERRIDES.get(m.group(1), m.group(1))


def parse_update(title: str, body: str) -> tuple[str, str, str]:
    title = title.strip()
    if re.search(r"(?i)\bthe\s+\S+\s+group\b|\bupdates?\s+in\s+the\b|\band\s+\d+\s+other", title):
        return "", "", ""  # grouped update
    for pat in TITLE_PATTERNS:
        m = pat.search(title)
        if m:
            return m.group("dep"), m.group("old").rstrip("."), m.group("new").rstrip(".")
    m = RENOVATE_TITLE.search(title)
    if m:
        cells = RENOVATE_BODY.findall(body or "")
        news = {n for _, n in cells}
        if len(cells) >= 1 and len(news) == 1 and m.group("new").lstrip("v") in news:
            return m.group("dep"), cells[0][0], m.group("new").lstrip("v")
    return "", "", ""


def latest_checks(gh: Github, repo: str, sha: str) -> dict[str, tuple[str, str]]:
    """name -> (status, conclusion) of the newest check run / commit status per name."""
    runs = (gh.get(f"{API}/repos/{repo}/commits/{sha}/check-runs?per_page=100") or {}).get("check_runs", [])
    statuses = (gh.get(f"{API}/repos/{repo}/commits/{sha}/status?per_page=100") or {}).get("statuses", [])

    latest: dict[str, tuple[str, str, str]] = {}  # name -> (timestamp, state, conclusion)
    for r in runs:
        name = r.get("name", "?")
        stamp = r.get("completed_at") or r.get("started_at") or ""
        if name not in latest or stamp >= latest[name][0]:
            latest[name] = (stamp, r.get("status", ""), r.get("conclusion") or "")
    for s in statuses:  # already newest first per context
        ctx = "status:" + s.get("context", "?")
        if ctx not in latest:
            state = s.get("state", "")
            latest[ctx] = (s.get("updated_at", ""), "completed" if state != "pending" else "pending",
                           {"success": "success", "failure": "failure", "error": "failure"}.get(state, ""))
    return {n: (st, c) for n, (_, st, c) in latest.items()}


def ci_status(gh: Github, repo: str, sha: str) -> tuple[str, list[str]]:
    """GREEN / NOT_GREEN / PENDING / NO_CI / NO_BUILD_CHECK for one commit, plus the check names."""
    latest = latest_checks(gh, repo, sha)
    if not latest:
        return "NO_CI", []
    names = sorted(latest)
    if any(st != "completed" for st, _ in latest.values()):
        return "PENDING", names
    if any(c in BAD_CONCLUSIONS or c == "" for _, c in latest.values()):
        return "NOT_GREEN", names
    build_like = [n for n, (_, c) in latest.items() if c == "success" and not TRIVIAL_CHECK.search(n)]
    if not build_like:
        return "NO_BUILD_CHECK", names
    return "GREEN", names


BOT_LOGINS = {"dependabot[bot]", "dependabot-preview[bot]", "renovate[bot]"}


def list_bot_prs(gh: Github, repo: str, max_pages: int = 6) -> list[dict]:
    """Merged bot PRs, newest first, from the pulls API (core quota; the search API's
    secondary rate limits are too tight for ~50 repos)."""
    items = []
    for page in range(1, max_pages + 1):
        prs = gh.get(f"{API}/repos/{repo}/pulls?state=closed&per_page=100&page={page}"
                     "&sort=created&direction=desc") or []
        items.extend(p for p in prs if p.get("merged_at") and (p.get("user") or {}).get("login") in BOT_LOGINS)
        if len(prs) < 100:
            break
    return items


def canonical_repo(gh: Github, repo: str) -> str:
    """Follow renames/transfers (e.g. alphagov/pay-* -> govuk-pay/pay-*); '' if gone."""
    info = gh.get(f"{API}/repos/{repo}") or {}
    return info.get("full_name", "")


def resolve_group(gh: Github, artifact: str, version: str, files: list[dict]) -> str:
    """groupId for a title that only names the artifactId.

    1. the <groupId> right before <artifactId>artifact</artifactId> in the pom.xml patch;
    2. otherwise Maven Central search, if exactly one groupId publishes artifact:version.
    """
    for f in files:
        text = "\n".join(line[1:] for line in (f.get("patch") or "").splitlines())
        m = re.search(r"<groupId>\s*([^<$]+?)\s*</groupId>\s*<artifactId>\s*" + re.escape(artifact)
                      + r"\s*</artifactId>", text)
        if m:
            return m.group(1)
    q = urllib.parse.urlencode({"q": f'a:"{artifact}" AND v:"{version}"', "rows": 20, "wt": "json"})
    try:
        res = gh.get(f"https://search.maven.org/solrsearch/select?{q}") or {}
    except (OSError, http.client.HTTPException):  # search.maven.org is often slow; skip the PR
        return ""
    groups = {d.get("g") for d in res.get("response", {}).get("docs", [])}
    return groups.pop() if len(groups) == 1 else ""


def inspect_pr(gh: Github, repo: str, item: dict, known: set) -> tuple[dict | None, str]:
    dep, old, new = parse_update(item.get("title", ""), item.get("body") or "")
    if not dep:
        return None, "title is not a single Maven coordinate update"
    pr = gh.get(f"{API}/repos/{repo}/pulls/{item['number']}")
    if not pr or not pr.get("merged_at"):
        return None, "not merged"
    if pr.get("commits", 0) > 5:
        return None, f"{pr.get('commits')} commits"
    files = gh.get(f"{API}/repos/{repo}/pulls/{item['number']}/files?per_page=100") or []
    names = [f.get("filename", "") for f in files]
    if not names or any(Path(n).name != "pom.xml" for n in names):
        return None, "changes files other than pom.xml: " + "|".join(names[:5])
    if ":" in dep:
        group, artifact = dep.split(":", 1)
    else:
        artifact = dep
        group = resolve_group(gh, artifact, new, files)
        if not group:
            return None, "groupId not resolvable"
    if (repo, group, artifact, old, new) in known:
        return None, "already in dataset"
    head = pr["head"]["sha"]
    status, checks = ci_status(gh, repo, head)
    if status != "GREEN":
        return None, f"head CI {status}"
    # Pre-update state = merge base of the PR. The PR's file list is computed against it and
    # is pom.xml-only, so post = pre + the dependency bump even if the base branch was
    # merged into the PR branch (Dependabot rebases, "Merge branch main" commits).
    cmp = gh.get(f"{API}/repos/{repo}/compare/{pr['base']['sha']}...{head}") or {}
    pre = (cmp.get("merge_base_commit") or {}).get("sha", "")
    if not pre:
        return None, "merge base not found"
    pre_status, _ = ci_status(gh, repo, pre)
    if pre_status == "NOT_GREEN":
        return None, "pre-update commit CI NOT_GREEN"
    bot = "dependabot" if "dependabot" in pr["user"]["login"] else "renovate"
    return {
        "project": repo.split("/")[1],
        "repo_url": pr["html_url"],
        "dependency_group_id": group,
        "dependency_artifact_id": artifact,
        "old_version": old,
        "new_version": new,
        "java_version": "",
        "failure_category": "",
        "pre_update_status": "PASS" if pre_status == "GREEN" else "UNKNOWN",
        "post_update_status": "PASS",
        "classification": "SAFE",
        "source_json": "",
        "label_source": "github_ci",
        "github_repo": repo,
        "pr_number": pr["number"],
        "bot": bot,
        "pre_commit_sha": pre,
        "post_commit_sha": head,
        "merge_commit_sha": pr.get("merge_commit_sha") or "",
        "merged_at": pr["merged_at"],
        "changed_files": "|".join(names),
        "ci_checks": "|".join(checks),
        "pre_ci_status": pre_status,
    }, "accepted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", type=int, default=45)
    ap.add_argument("--per-repo", type=int, default=2)
    ap.add_argument("--max-inspect-per-repo", type=int, default=25)
    args = ap.parse_args()
    if "Authorization" not in HEADERS:
        print("Set GITHUB_TOKEN (e.g. GITHUB_TOKEN=$(gh auth token)); the anonymous rate limit is too low.")
        sys.exit(1)

    rows = list(csv.DictReader(DATASET.open(encoding="utf-8-sig")))
    known = {(repo_of(r["repo_url"]), r["dependency_group_id"], r["dependency_artifact_id"],
              r["old_version"], r["new_version"]) for r in rows}
    repos = sorted({repo_of(r["repo_url"]) for r in rows if r["classification"] == "BREAKING"} - SKIP_REPOS - {""})
    print(f"Mining {len(repos)} repositories that have BREAKING samples")

    gh = Github()
    candidates: dict[str, list[dict]] = {}
    rejections = []
    for i, repo in enumerate(repos, 1):
        accepted: list[dict] = []
        canon = canonical_repo(gh, repo)
        if not canon:
            print(f"[{i}/{len(repos)}] {repo}: repository not found, skipped")
            continue
        repo = canon
        items = list_bot_prs(gh, repo)
        inspected = 0
        for item in items:
            if len(accepted) >= args.per_repo + 2 or inspected >= args.max_inspect_per_repo:
                break
            dep, _, _ = parse_update(item.get("title", ""), item.get("body") or "")
            if dep:
                inspected += 1
            try:
                row, reason = inspect_pr(gh, repo, item, known)
            except urllib.error.HTTPError as exc:
                row, reason = None, f"HTTP {exc.code}"
            except (OSError, http.client.HTTPException) as exc:
                row, reason = None, f"network error: {exc}"
            if row and any(r["dependency_artifact_id"] == row["dependency_artifact_id"] for r in accepted):
                row, reason = None, "same artifact already accepted for this repo"
            if row:
                accepted.append(row)
            rejections.append({"github_repo": repo, "pr_url": item.get("html_url", ""),
                               "title": item.get("title", ""), "result": reason})
        candidates[repo] = accepted
        print(f"[{i}/{len(repos)}] {repo}: {len(items)} bot PRs, {len(accepted)} accepted")
        gh.save()  # keep progress if a later repo crashes
    gh.save()

    # Round-robin over repos so no single project dominates the SAFE class.
    selected: list[dict] = []
    for k in range(args.per_repo):
        for repo in candidates:
            if len(selected) >= args.target:
                break
            if len(candidates.get(repo, [])) > k:
                selected.append(candidates[repo][k])

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(selected)
    OUT_JSON.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    with REJECT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["github_repo", "pr_url", "title", "result"])
        w.writeheader()
        w.writerows(rejections)

    combined = []
    for r in rows:
        combined.append({**{k: r.get(k, "") for k in FIELDS}, "label_source": "bump_docker_reproduction",
                         "github_repo": repo_of(r["repo_url"])})
    combined.extend(selected)
    with COMBINED_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(combined)

    print(f"\nSelected {len(selected)} SAFE samples from {len({r['github_repo'] for r in selected})} repos")
    print("Bots:", dict(Counter(r["bot"] for r in selected)))
    print("Pre-update CI:", dict(Counter(r["pre_ci_status"] for r in selected)))
    print("Rejection reasons:", Counter(re.sub(r":.*", "", r["result"]) for r in rejections).most_common(10))
    print(f"Combined dataset: {Counter(r['classification'] for r in combined)} -> {COMBINED_CSV.name}")


if __name__ == "__main__":
    main()
