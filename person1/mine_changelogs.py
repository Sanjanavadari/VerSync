from __future__ import annotations

import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "changelogs"
CACHE = ROOT / "safe_pr_api_cache.json"
BUMP_BENCH = ROOT.parent / "dataset" / "bump" / "data" / "benchmark"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "VerSync-Changelog-Miner",
    "X-GitHub-Api-Version": "2022-11-28",
}
DETAIL_RE = re.compile(r"<details\b[^>]*>(.*?)</details\s*>", re.I | re.S)
SUMMARY_RE = re.compile(r"<summary\b[^>]*>(.*?)</summary\s*>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
MD_HEADING_RE = re.compile(r"(?im)^\s{0,3}#{1,6}\s*(release notes?|changelog|breaking changes?)\s*#*\s*$")


def api_get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_cache() -> dict:
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def strip_html(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(?:p|div|li|h[1-6])\s*>", "\n", text)
    text = TAG_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def sections_from_pr_body(body: str) -> dict[str, str]:
    """Extract relevant HTML details and explicitly titled Markdown sections."""
    out: dict[str, str] = {}
    if not body:
        return out
    for match in DETAIL_RE.finditer(body):
        block = match.group(1)
        summary = SUMMARY_RE.search(block)
        if not summary:
            continue
        title = strip_html(summary.group(1)).lower()
        if "release note" in title:
            key = "release_notes"
        elif "changelog" in title:
            key = "changelog"
        elif "breaking change" in title:
            key = "breaking_changes"
        elif title.strip() == "commits":
            key = "commits"
        else:
            continue
        text = strip_html(block[summary.end():])[:12000]
        if text:
            out[key] = text

    headings = list(MD_HEADING_RE.finditer(body))
    for i, match in enumerate(headings):
        title = match.group(1).lower()
        key = "release_notes" if "release" in title else "changelog" if "changelog" in title else "breaking_changes"
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        level_match = re.match(r"\s*(#+)", match.group(0))
        level = len(level_match.group(1)) if level_match else 1
        next_heading = re.search(rf"(?m)^#{{1,{level}}}\s+\S.*$", body[match.end():end])
        if next_heading:
            end = match.end() + next_heading.start()
        text = strip_html(body[match.end():end])[:12000]
        if text:
            out.setdefault(key, text)
    return out


def harvest_from_cache(cache: dict) -> dict[str, dict]:
    found = {}
    for url, payload in cache.items():
        if not isinstance(payload, dict):
            continue
        html_url = (payload.get("html_url") or url).rstrip("/")
        if "/pull/" not in html_url and not payload.get("pull_request"):
            continue
        sections = sections_from_pr_body(payload.get("body") or "")
        if sections and html_url:
            found[html_url] = {"pr_url": html_url, "title": payload.get("title", ""), "sources": ["pr_body"], "sections": sections}
    return found


def bump_samples() -> list[dict]:
    rows = []
    if not BUMP_BENCH.exists():
        return rows
    for path in sorted(BUMP_BENCH.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dep = data.get("updatedDependency") or {}
        rows.append({
            "project": data.get("project", ""), "pr_url": data.get("url", ""),
            "group_id": dep.get("dependencyGroupID", ""), "artifact_id": dep.get("dependencyArtifactID", ""),
            "old_version": dep.get("previousVersion", ""), "new_version": dep.get("newVersion", ""),
            "github_compare": dep.get("githubCompareLink", ""), "github_repo": dep.get("githubRepoSlug", ""),
            "maven_source_link": dep.get("mavenSourceLink", "") or dep.get("mavenSourceLinkURL", ""),
            "source_json": path.name, "failure_category": data.get("failureCategory", ""),
        })
    return rows


def fetch_github_release_notes(repo_slug: str, version: str) -> str:
    if not repo_slug or "/" not in repo_slug or " " in repo_slug:
        return ""
    tags = list(dict.fromkeys([version, f"v{version}", f"{version}.RELEASE", f"v{version}.RELEASE"]))
    for tag in tags:
        url = f"https://api.github.com/repos/{repo_slug}/releases/tags/{urllib.parse.quote(tag, safe='')}"
        try:
            data = api_get(url)
            body = data.get("body") or "" if isinstance(data, dict) else ""
            if body.strip():
                return body.strip()[:12000]
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise
        except (urllib.error.URLError, json.JSONDecodeError):
            return ""
        time.sleep(0.15)
    return ""


def fetch_changelog_md(repo_slug: str) -> tuple[str, str]:
    """Try common changelog filenames on the repository default branch, main and master."""
    if not repo_slug or "/" not in repo_slug or " " in repo_slug:
        return "", ""
    branches = ["main", "master"]
    try:
        repo = api_get(f"https://api.github.com/repos/{repo_slug}")
        default = repo.get("default_branch") if isinstance(repo, dict) else None
        if default:
            branches = list(dict.fromkeys([default, *branches]))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        pass
    paths = ["CHANGELOG.md", "CHANGES.md", "CHANGELOG.adoc", "History.md", "docs/CHANGELOG.md", "changelog.txt", "CHANGELOG", "CHANGES"]
    for branch in branches:
        for path in paths:
            url = f"https://raw.githubusercontent.com/{repo_slug}/{urllib.parse.quote(branch, safe='')}/{path}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "VerSync-Changelog-Miner"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    text = resp.read().decode("utf-8", errors="replace").strip()
                    if text:
                        return text[:15000], url
            except (urllib.error.HTTPError, urllib.error.URLError):
                continue
            time.sleep(0.08)
    return "", ""


def safe_name(*parts: str) -> str:
    raw = "__".join(str(p or "") for p in parts)
    return re.sub(r"[^\w.\-]+", "_", raw)[:180]


def has_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pr_changelogs = harvest_from_cache(load_cache())
    print(f"PR bodies with extracted sections in cache: {len(pr_changelogs)}")

    safe_csv = ROOT / "safe_pr_java_candidates.csv"
    if not safe_csv.exists():
        safe_csv = ROOT / "safe_pr_candidates.csv"
    safe_rows = []
    if safe_csv.exists():
        with safe_csv.open(newline="", encoding="utf-8-sig") as f:
            safe_rows = list(csv.DictReader(f))

    index = []
    for row in safe_rows:
        pr_url = (row.get("pr_url") or "").rstrip("/")
        dep, old, new = row.get("dependency_name", ""), row.get("old_version", ""), row.get("new_version", "")
        mined = pr_changelogs.get(pr_url, {})
        sections = dict(mined.get("sections") or {})
        sources = list(mined.get("sources") or [])
        text_found = any(has_text(v) for v in sections.values())
        record = {
            "kind": "safe_candidate", "project": row.get("project", ""), "pr_url": row.get("pr_url", ""),
            "dependency": dep, "old_version": old, "new_version": new, "sources": sources,
            "sections": sections, "github_compare": "",
            "evidence_status": "TEXT_EXTRACTED" if text_found else "NO_TEXT_FOUND",
        }
        out_path = OUT_DIR / f"{safe_name(record['project'], dep.replace(':', '_'), f'{old}_to_{new}')}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({
            "kind": record["kind"], "project": record["project"], "dependency": dep, "old_version": old,
            "new_version": new, "pr_url": row.get("pr_url", ""),
            "has_release_notes": "YES" if has_text(sections.get("release_notes")) else "NO",
            "has_changelog": "YES" if any(has_text(sections.get(k)) for k in ("changelog", "changelog_md")) else "NO",
            "has_compare_link": "NO", "evidence_status": record["evidence_status"],
            "sources": "|".join(sources) or "none", "json_path": str(out_path),
        })

    bump = bump_samples()
    print(f"BUMP benchmark samples: {len(bump)}")
    rate_limited = False
    for i, row in enumerate(bump[:80]):
        pr_url = (row["pr_url"] or "").rstrip("/")
        mined = pr_changelogs.get(pr_url, {})
        sections = dict(mined.get("sections") or {})
        sources = list(mined.get("sources") or [])
        compare, maven_source = row["github_compare"] or "", row["maven_source_link"] or ""
        if compare:
            sections["github_compare_link"] = compare
        if maven_source:
            sections["maven_source_link"] = maven_source

        repo = row.get("github_repo") or ""
        if repo and not rate_limited and not has_text(sections.get("release_notes")):
            try:
                notes = fetch_github_release_notes(repo, row["new_version"])
                if notes:
                    sections["release_notes"] = notes
                    sources.append("github_release")
                else:
                    changelog, changelog_url = fetch_changelog_md(repo)
                    if changelog:
                        sections["changelog_md"] = changelog
                        sections["changelog_url"] = changelog_url
                        sources.append("changelog_md")
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 429):
                    print(f"GitHub rate limited at BUMP sample {i + 1}; skipping further remote fetches.")
                    rate_limited = True

        text_found = any(has_text(v) for k, v in sections.items() if k not in ("github_compare_link", "maven_source_link", "changelog_url"))
        evidence_status = "TEXT_EXTRACTED" if text_found else "LINK_ONLY" if compare or maven_source else "NO_TEXT_FOUND"
        record = {
            "kind": "bump_breaking", "project": row["project"], "pr_url": row["pr_url"],
            "dependency": f"{row['group_id']}:{row['artifact_id']}", "old_version": row["old_version"],
            "new_version": row["new_version"], "failure_category": row["failure_category"],
            "github_compare": compare, "maven_source_link": maven_source,
            "sources": list(dict.fromkeys(sources + (["bump_compare_link"] if compare else []) + (["maven_source_link"] if maven_source else []))),
            "sections": sections, "evidence_status": evidence_status, "source_json": row["source_json"],
        }
        version_pair = f"{row['old_version']}_to_{row['new_version']}"
        out_path = OUT_DIR / f"{safe_name(row['project'], row['artifact_id'], version_pair)}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({
            "kind": record["kind"], "project": row["project"], "dependency": record["dependency"],
            "old_version": row["old_version"], "new_version": row["new_version"], "pr_url": row["pr_url"],
            "has_release_notes": "YES" if has_text(sections.get("release_notes")) else "NO",
            "has_changelog": "YES" if any(has_text(sections.get(k)) for k in ("changelog", "changelog_md")) else "NO",
            "has_compare_link": "YES" if bool(compare) else "NO", "evidence_status": evidence_status,
            "sources": "|".join(record["sources"]) or "none", "json_path": str(out_path),
        })
        if i and i % 20 == 0:
            print(f"  processed {i} BUMP samples...")

    index_path = OUT_DIR / "changelog_index.csv"
    fields = ["kind", "project", "dependency", "old_version", "new_version", "pr_url", "has_release_notes", "has_changelog", "has_compare_link", "evidence_status", "sources", "json_path"]
    with index_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index)

    print("\n========== SUMMARY ==========")
    print("Changelog records:", len(index))
    print("Records with extracted text:", sum(r["evidence_status"] == "TEXT_EXTRACTED" for r in index))
    print("Records with links only:", sum(r["evidence_status"] == "LINK_ONLY" for r in index))
    print("Records with no text found:", sum(r["evidence_status"] == "NO_TEXT_FOUND" for r in index))
    print("Index:", index_path)


if __name__ == "__main__":
    main()
