"""
Task 3 (per dataset sample): changelog / release-note text for every update in
combined_dataset.csv (falls back to final_verified_dataset.csv), SAFE and BREAKING alike.

mine_changelogs.py covers the first 80 BUMP benchmark files plus a list of
safe candidates, so most dataset samples were LINK_ONLY or missing, and none of
the SAFE samples had an entry. A feature that exists only for BREAKING samples
lets a model read the label from "has a changelog", so this script treats both
classes identically.

Where the text comes from (all sources are collected, not just the first hit):
  1. pr_body:      the Dependabot/Renovate PR body (release notes, changelog and
                   commit sections the bot copies in), when the sample has a PR URL.
  2. github_release: every GitHub release of the library whose tag version is in
                   (old_version, new_version].
  3. changelog_file: the part of CHANGELOG.md / CHANGES.md / RELEASE-NOTES.md on the
                   default branch between the new-version and old-version headings.

The library's source repository is taken from the new version's POM <scm> /
<url> (walking up parent POMs), with Apache gitbox URLs mapped to the
github.com/apache mirror. BUMP's githubRepoSlug is used only as a fallback,
because it is sometimes wrong (mina-core -> apache/mina-sshd instead of apache/mina).

Usage:
  GITHUB_TOKEN=$(gh auth token) python mine_sample_changelogs.py

Outputs:
  changelogs/samples/<project>__<artifact>__<old>_to_<new>.json
  changelogs/sample_changelog_index.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from api_diff_extractor import download, rel
from mine_changelogs import sections_from_pr_body, strip_html
from mine_safe_ci_samples import API, Github

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "combined_dataset.csv"
if not DATASET.exists():
    DATASET = ROOT / "final_verified_dataset.csv"
OUT_DIR = ROOT / "changelogs" / "samples"
INDEX = ROOT / "changelogs" / "sample_changelog_index.csv"
BUMP_BENCH = ROOT.parent / "dataset" / "bump" / "data" / "benchmark"
MAX_TEXT = 30000
CHANGELOG_FILES = ["CHANGELOG.md", "CHANGES.md", "RELEASE-NOTES.md", "RELEASE_NOTES.md", "HISTORY.md",
                   "History.md", "NEWS.md", "CHANGELOG.adoc", "CHANGELOG.txt", "CHANGES.txt",
                   "src/changes/changes.xml", "docs/CHANGELOG.md"]
GITHUB_URL = re.compile(r"github\.com[/:]([\w.\-]+)/([\w.\-]+?)(?:\.git)?(?:[/#?]|$)")
APACHE_GIT = re.compile(r"(?:gitbox|git-wip-us|git)\.apache\.org/repos/asf(?:\?p=|/)([\w.\-]+?)(?:\.git)?(?:[;/]|$)")


def vkey(v: str) -> tuple:
    """Sortable version key: numeric parts compare numerically, qualifiers (-M1, -rc2) sort before the release."""
    parts = re.split(r"[.\-_+]", v.lower().lstrip("v"))
    key = []
    for p in parts:
        if p.isdigit():
            key.append((1, int(p), ""))
        elif p in ("final", "release", "ga"):
            continue
        else:
            key.append((0, 0, p))
    key.append((0, 1, ""))  # end marker: above any qualifier, below any further number
    return tuple(key)


def tag_version(tag: str) -> str:
    """Version embedded in a tag: v1.2.3, mina-2.2.1, rel/commons-io-2.13.0, release_1_2, 1500.v1a2b3c."""
    text = tag.replace("/", " ").replace("_", ".")
    m = re.findall(r"(?<![\w])v?(\d+(?:\.\d+)+(?:[.\-](?:[A-Za-z]+\d*|\d+))*)", text)
    if m:
        return m[-1]
    m = re.search(r"(?<![\w.])(\d+(?:\.\d+)*\.v[0-9a-f]+)", text)  # Jenkins incrementals
    return m.group(1) if m else ""


def in_range(v: str, old: str, new: str) -> bool:
    try:
        return vkey(old) < vkey(v) <= vkey(new)
    except TypeError:
        return False


def source_repo_from_pom(group: str, artifact: str, version: str, depth: int = 0) -> str:
    try:
        root = ET.fromstring(download(group, artifact, version, "pom").read_bytes())
    except Exception:
        return ""
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    for path in ("scm/url", "scm/connection", "scm/developerConnection", "url", "issueManagement/url"):
        text = (root.findtext(path) or "").strip()
        m = GITHUB_URL.search(text)
        if m and "${" not in text:
            return f"{m.group(1)}/{m.group(2)}"
        m = APACHE_GIT.search(text)
        if m:
            return f"apache/{m.group(1)}"
    parent = root.find("parent")
    if parent is not None and depth < 4:
        return source_repo_from_pom(parent.findtext("groupId", ""), parent.findtext("artifactId", ""),
                                    parent.findtext("version", ""), depth + 1)
    return ""


def bump_slug(source_json: str) -> str:
    path = BUMP_BENCH / source_json if source_json else None
    if not path or not path.exists():
        return ""
    dep = json.loads(path.read_text(encoding="utf-8")).get("updatedDependency", {})
    return dep.get("githubRepoSlug") or ""


def releases_in_range(gh: Github, repo: str, old: str, new: str) -> list[dict]:
    out = []
    for page in (1, 2, 3):
        rels = gh.get(f"{API}/repos/{repo}/releases?per_page=100&page={page}") or []
        for r in rels:
            v = tag_version(r.get("tag_name") or "") or tag_version(r.get("name") or "")
            if v and in_range(v, old, new) and (r.get("body") or "").strip():
                out.append({"tag": r["tag_name"], "version": v, "body": r["body"].strip()})
        if len(rels) < 100:
            break
    return sorted(out, key=lambda r: vkey(r["version"]))


def changelog_section(gh: Github, repo: str, old: str, new: str) -> tuple[str, str]:
    """Text between the heading for `new` (or the newest in-range version) and the heading for `old`."""
    info = gh.get(f"{API}/repos/{repo}") or {}
    branch = info.get("default_branch") or "master"
    for name in CHANGELOG_FILES:
        url = f"https://raw.githubusercontent.com/{repo}/{urllib.parse.quote(branch)}/{name}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "VerSync"}),
                                        timeout=20) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
        lines = text.splitlines()
        # A version heading: markdown/adoc heading, "Version x", "x (date)", or <release version="x">.
        heads = []
        for i, line in enumerate(lines):
            if re.match(r"^\s*(#{1,4}|={1,4}|\*\*|version\b|release\b|<release\b|\[?v?\d+\.\d+)", line, re.I):
                v = tag_version(line)
                if v:
                    heads.append((i, v))
        start = next((i for i, v in heads if in_range(v, old, new)), None)
        if start is None:
            continue
        end = next((i for i, v in heads if i > start and vkey(v) <= vkey(old)), len(lines))
        section = "\n".join(lines[start:end]).strip()
        if section:
            return section[:MAX_TEXT], url
    return "", ""


def safe_name(*parts: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", "__".join(parts))[:180]


def main() -> None:
    rows = list(csv.DictReader(DATASET.open(encoding="utf-8-sig")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gh = Github()
    index = []
    repo_cache: dict[tuple, str] = {}
    for i, row in enumerate(rows, 1):
        g, a = row["dependency_group_id"], row["dependency_artifact_id"]
        old, new = row["old_version"], row["new_version"]
        project = row["project"]
        print(f"[{i}/{len(rows)}] {row['classification']:8s} {project} {g}:{a} {old}->{new}")

        if (g, a, new) not in repo_cache:
            repo_cache[(g, a, new)] = source_repo_from_pom(g, a, new)
        pom_repo = repo_cache[(g, a, new)]
        slug = bump_slug(row.get("source_json", ""))
        repo, origin = (pom_repo, "pom_scm") if pom_repo else (slug, "bump_slug") if slug else ("", "")
        if repo and not (gh.get(f"{API}/repos/{repo}") or {}).get("full_name"):
            repo, origin = (slug, "bump_slug") if slug and slug != repo else ("", "")
        elif repo:
            repo = gh.get(f"{API}/repos/{repo}")["full_name"]  # follow renames

        sections: dict[str, str] = {}
        sources: list[str] = []
        pr = re.match(r"https://github\.com/([^/]+/[^/]+)/pull/(\d+)", row["repo_url"])
        if pr:
            data = gh.get(f"{API}/repos/{pr.group(1)}/pulls/{pr.group(2)}") or {}
            for k, v in sections_from_pr_body(data.get("body") or "").items():
                sections[f"pr_body_{k}"] = v
            if any(k.startswith("pr_body_") and k != "pr_body_commits" for k in sections):
                sources.append("pr_body")

        releases = releases_in_range(gh, repo, old, new) if repo else []
        if releases:
            sections["release_notes"] = "\n\n".join(f"## {r['tag']}\n{strip_html(r['body'])}"
                                                    for r in releases)[:MAX_TEXT]
            sources.append("github_release")
        section, section_url = changelog_section(gh, repo, old, new) if repo else ("", "")
        if section:
            sections["changelog_file"] = section
            sources.append("changelog_file")

        text_chars = sum(len(v) for k, v in sections.items() if k != "pr_body_commits")
        status = ("TEXT_EXTRACTED" if text_chars else
                  "COMMITS_ONLY" if sections.get("pr_body_commits") else
                  "LINK_ONLY" if repo else "MISSING")
        compare = f"https://github.com/{repo}/compare/{old}...{new}" if repo else ""
        record = {
            "project": project, "classification": row["classification"],
            "label_source": row.get("label_source", "bump_docker_reproduction"),
            "pr_url": row["repo_url"], "dependency": f"{g}:{a}", "old_version": old, "new_version": new,
            "source_repo": repo, "source_repo_origin": origin, "bump_repo_slug": slug,
            "releases_in_range": [r["tag"] for r in releases], "changelog_url": section_url,
            "github_compare": compare, "sources": sources, "sections": sections,
            "evidence_status": status,
        }
        out = OUT_DIR / f"{safe_name(project, a, f'{old}_to_{new}')}.json"
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({
            "project": project, "classification": row["classification"], "label_source": record["label_source"],
            "dependency": f"{g}:{a}", "old_version": old, "new_version": new, "pr_url": row["repo_url"],
            "source_repo": repo, "source_repo_origin": origin,
            "bump_slug_disagrees": "YES" if slug and repo and slug.lower() != repo.lower() else "",
            "has_pr_body_notes": "YES" if "pr_body" in sources else "NO",
            "n_releases_in_range": len(releases), "has_changelog_section": "YES" if section else "NO",
            "text_chars": text_chars, "evidence_status": status, "json_path": rel(out),
        })
        print(f"    {status} repo={repo or '-'} ({origin}) sources={sources}")
    gh.save()

    with INDEX.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print("\nEvidence status:", dict(Counter(r["evidence_status"] for r in index)))
    print("By class:", dict(Counter((r["classification"], r["evidence_status"]) for r in index)))
    print("Repo origin:", dict(Counter(r["source_repo_origin"] or "none" for r in index)))
    print("Index:", rel(INDEX))


if __name__ == "__main__":
    main()
