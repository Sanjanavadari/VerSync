"""
Task 2: API-diff extractor — compare public methods in old vs new library JARs.

Downloads artifacts from Maven Central or the Jenkins repository (or accepts
local paths), runs `javap -public`, and reports removed / changed / added
method signatures plus removed / added classes.

Packaging is read from the artifact's POM first:
  * jar / maven-plugin / bundle ...: the JAR is diffed.
  * hpi / jpi (Jenkins plugins): the .hpi is diffed, including the libraries it
    bundles in WEB-INF/lib. Wrapper plugins such as github-api or
    commons-text-api only exist to ship a library, so that library's API is
    what the consumer actually calls.
  * pom (BOMs, parent POMs): there is no API to diff. Instead the managed /
    declared dependency versions of the two POMs are compared
    (pom_dependency_changes), which is what a BOM update actually changes.
For every artifact the direct dependencies declared in the two POMs are also
compared, because a release that drops or upgrades a transitive dependency
can break consumers without any method changing.

Usage:
  python api_diff_extractor.py --group org.slf4j --artifact slf4j-api \\
      --old 1.7.36 --new 2.0.3

  python api_diff_extractor.py --csv combined_dataset.csv [--limit 10]

  python api_diff_extractor.py --old-jar path/old.jar --new-jar path/new.jar
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "jar_cache"
OUT_DIR = ROOT / "api_diffs"
REPOSITORIES = [
    "https://repo1.maven.org/maven2",
    "https://repo.jenkins-ci.org/public",
]
NO_API_PACKAGING = {"pom"}
PLUGIN_PACKAGING = {"hpi", "jpi"}

METHOD_RE = re.compile(
    r"^\s*(?:public|protected)\s+(?:static\s+|final\s+|native\s+|synchronized\s+|abstract\s+|default\s+)*"
    r"(?:<[^>]*>\s+)?(?P<ret>[\w.<>,?\[\]\s]+?)\s+(?P<name>[\w$]+)\s*\((?P<params>[^)]*)\)"
)
CTOR_RE = re.compile(
    r"^\s*(?:public|protected)\s+(?P<name>[\w$]+)\s*\((?P<params>[^)]*)\)"
)


def rel(path: Path) -> str:
    """Path relative to person1/ so outputs are portable across machines."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def artifact_path(group: str, artifact: str, version: str, ext: str) -> str:
    gpath = group.replace(".", "/")
    return f"{gpath}/{artifact}/{version}/{artifact}-{version}.{ext}"


def maven_url(group: str, artifact: str, version: str, ext: str = "jar",
              repo: str = REPOSITORIES[0]) -> str:
    return f"{repo}/{artifact_path(group, artifact, version, ext)}"


def download(group: str, artifact: str, version: str, ext: str) -> Path:
    """Download from the first repository that has the file (cached in jar_cache/)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{group}_{artifact}_{version}.{ext}"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    errors = []
    for repo in REPOSITORIES:
        url = maven_url(group, artifact, version, ext, repo)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  Downloaded {url}")
            return dest
        except urllib.error.HTTPError as exc:
            errors.append(f"{exc.code} {url}")
    dest.unlink(missing_ok=True)
    raise FileNotFoundError("not found in any repository: " + "; ".join(errors))


def download_jar(group: str, artifact: str, version: str) -> Path:
    return download(group, artifact, version, "jar")


def _strip_ns(root: ET.Element) -> None:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def read_pom(group: str, artifact: str, version: str) -> dict:
    """Packaging, declared dependencies and managed dependencies of one POM."""
    path = download(group, artifact, version, "pom")
    root = ET.fromstring(path.read_bytes())
    _strip_ns(root)
    props = {p.tag: (p.text or "").strip() for p in root.findall("properties/*")}
    props.update({"project.version": version, "version": version})

    def resolve(v: str) -> str:
        return re.sub(r"\$\{([^}]+)\}", lambda m: props.get(m.group(1), m.group(0)), v or "")

    def deps(xpath: str) -> dict[str, str]:
        out = {}
        for d in root.findall(xpath):
            g = resolve(d.findtext("groupId", "")).strip()
            a = resolve(d.findtext("artifactId", "")).strip()
            if g and a:
                out[f"{g}:{a}"] = resolve(d.findtext("version", "")).strip()
        return out

    return {
        "packaging": (root.findtext("packaging") or "jar").strip(),
        "dependencies": deps("dependencies/dependency"),
        "managed": deps("dependencyManagement/dependencies/dependency"),
    }


def diff_dependency_maps(old: dict[str, str], new: dict[str, str]) -> dict:
    return {
        "removed": sorted(k for k in old if k not in new),
        "added": sorted(k for k in new if k not in old),
        "version_changed": [
            {"dependency": k, "old": old[k], "new": new[k]}
            for k in sorted(set(old) & set(new)) if old[k] != new[k]
        ],
    }


def list_class_names(jar_path: Path) -> list[str]:
    names = []
    with zipfile.ZipFile(jar_path) as zf:
        for info in zf.infolist():
            name = info.filename
            # Skip module descriptors and multi-release duplicates under META-INF/versions/.
            if (name.endswith(".class") and not name.endswith("module-info.class")
                    and not name.startswith("META-INF/")):
                names.append(name[:-6].replace("/", "."))
    return names


def parse_javap_batch(javap_out: str) -> tuple[dict[str, str], set[str]]:
    """Parse `javap` output that may contain multiple class dumps."""
    methods: dict[str, str] = {}
    classes: set[str] = set()
    current_class = ""
    simple = ""
    for line in javap_out.splitlines():
        header = re.match(r"^(?:public\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:sealed\s+|non-sealed\s+)?"
                          r"(?:abstract\s+)?(?:class|interface|enum|@interface|record)\s+([\w.$]+)", line)
        if header:
            current_class = header.group(1)
            classes.add(current_class)
            simple = current_class.rsplit(".", 1)[-1].rsplit("$", 1)[-1]
            continue
        if not current_class:
            continue
        line = line.rstrip()
        c = CTOR_RE.match(line)
        if c and c.group("name") in (simple, current_class):
            params = re.sub(r"\s+", " ", c.group("params").strip())
            key = f"{current_class}#<init>({params})"
            methods[key] = f"{simple}({params})"
            continue
        m = METHOD_RE.match(line)
        if m:
            name = m.group("name")
            params = re.sub(r"\s+", " ", m.group("params").strip())
            ret = re.sub(r"\s+", " ", m.group("ret").strip())
            key = f"{current_class}#{name}({params})"
            methods[key] = f"{ret} {name}({params})"
    return methods, classes


def extract_api(jar_path: Path, batch_size: int = 40) -> tuple[dict[str, str], set[str]]:
    classes = [
        cls for cls in list_class_names(jar_path)
        if not re.search(r"\$\d+$", cls)
    ]
    api: dict[str, str] = {}
    public_classes: set[str] = set()
    for i in range(0, len(classes), batch_size):
        batch = classes[i:i + batch_size]
        try:
            proc = subprocess.run(
                ["javap", "-public", "-classpath", str(jar_path), *batch],
                capture_output=True,
                text=True,
                timeout=120,
                errors="replace",
            )
            m, c = parse_javap_batch(proc.stdout or "")
            api.update(m)
            public_classes |= c
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"    javap batch error at {i}: {exc}")
    return api, public_classes


def extract_api_any(path: Path) -> tuple[dict[str, str], set[str]]:
    """JAR directly; for .hpi plugins, the plugin's own classes plus every bundled WEB-INF/lib jar."""
    if path.suffix != ".hpi":
        return extract_api(path)
    api: dict[str, str] = {}
    classes: set[str] = set()
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as zf:
        inner = [n for n in zf.namelist() if n.startswith("WEB-INF/lib/") and n.endswith(".jar")]
        for name in inner:
            target = Path(tmp) / Path(name).name
            target.write_bytes(zf.read(name))
            m, c = extract_api(target)
            api.update(m)
            classes |= c
        if any(n.startswith("WEB-INF/classes/") and n.endswith(".class") for n in zf.namelist()):
            cls_jar = Path(tmp) / "_classes.jar"
            with zipfile.ZipFile(cls_jar, "w") as out:
                for n in zf.namelist():
                    if n.startswith("WEB-INF/classes/"):
                        out.writestr(n[len("WEB-INF/classes/"):], zf.read(n))
            m, c = extract_api(cls_jar)
            api.update(m)
            classes |= c
    return api, classes


def name_only(key: str) -> str:
    # class#name(params) -> class#name
    if "#" not in key:
        return key
    cls, rest = key.split("#", 1)
    name = rest.split("(", 1)[0]
    return f"{cls}#{name}"


def diff_apis(old_api: dict[str, str], new_api: dict[str, str]) -> dict:
    old_keys = set(old_api)
    new_keys = set(new_api)
    removed = sorted(old_keys - new_keys)
    added = sorted(new_keys - old_keys)

    # Changed: same class+method name, different descriptor
    old_by_name: dict[str, list[str]] = {}
    for k in removed:
        old_by_name.setdefault(name_only(k), []).append(k)
    new_by_name: dict[str, list[str]] = {}
    for k in added:
        new_by_name.setdefault(name_only(k), []).append(k)

    changed = []
    used_old = set()
    used_new = set()
    for n in sorted(set(old_by_name) & set(new_by_name)):
        o = old_by_name[n][0]
        nkey = new_by_name[n][0]
        changed.append({
            "name": n,
            "old": old_api[o],
            "new": new_api[nkey],
            "old_key": o,
            "new_key": nkey,
        })
        used_old.add(o)
        used_new.add(nkey)

    removed_clean = [k for k in removed if k not in used_old]
    added_clean = [k for k in added if k not in used_new]

    return {
        "removed": [{"key": k, "signature": old_api[k]} for k in removed_clean],
        "added": [{"key": k, "signature": new_api[k]} for k in added_clean],
        "changed": changed,
        "counts": {
            "old_methods": len(old_api),
            "new_methods": len(new_api),
            "removed": len(removed_clean),
            "added": len(added_clean),
            "changed": len(changed),
        },
    }


def compare_jars(old_jar: Path, new_jar: Path) -> dict:
    print(f"  Extracting API from {old_jar.name} ...")
    old_api, old_classes = extract_api_any(old_jar)
    print(f"  Extracting API from {new_jar.name} ...")
    new_api, new_classes = extract_api_any(new_jar)
    result = diff_apis(old_api, new_api)
    result["removed_classes"] = sorted(old_classes - new_classes)
    result["added_classes"] = sorted(new_classes - old_classes)
    result["counts"]["removed_classes"] = len(result["removed_classes"])
    result["counts"]["added_classes"] = len(result["added_classes"])
    result["old_jar"] = rel(old_jar)
    result["new_jar"] = rel(new_jar)
    return result


def compare_coords(group: str, artifact: str, old_v: str, new_v: str) -> dict:
    """Full comparison of one update; 'status' says whether an API diff was possible."""
    base = {"group_id": group, "artifact_id": artifact, "old_version": old_v, "new_version": new_v}
    old_pom = read_pom(group, artifact, old_v)
    new_pom = read_pom(group, artifact, new_v)
    pom_changes = {
        "dependencies": diff_dependency_maps(old_pom["dependencies"], new_pom["dependencies"]),
        "managed": diff_dependency_maps(old_pom["managed"], new_pom["managed"]),
    }
    packaging = new_pom["packaging"]
    if packaging in NO_API_PACKAGING:
        return {**base, "status": "NOT_APPLICABLE_POM", "packaging": packaging,
                "pom_dependency_changes": pom_changes,
                "counts": {"managed_version_changed": len(pom_changes["managed"]["version_changed"]),
                           "managed_removed": len(pom_changes["managed"]["removed"]),
                           "managed_added": len(pom_changes["managed"]["added"])}}
    ext = "hpi" if packaging in PLUGIN_PACKAGING else "jar"
    old_jar = download(group, artifact, old_v, ext)
    new_jar = download(group, artifact, new_v, ext)
    result = compare_jars(old_jar, new_jar)
    c = result["counts"]
    status = "OK" if (c["old_methods"] or c["new_methods"]) else "EMPTY_API"
    if status == "OK" and not (c["removed"] or c["added"] or c["changed"]):
        status = "OK_NO_METHOD_CHANGE"
    result.update(base)
    result.update({"status": status, "packaging": packaging, "pom_dependency_changes": pom_changes})
    return result


SUMMARY_FIELDS = [
    "project", "classification", "group_id", "artifact_id", "old_version", "new_version", "status",
    "packaging", "removed", "changed", "added", "removed_classes", "added_classes", "old_methods",
    "new_methods", "pom_deps_removed", "pom_deps_added", "pom_deps_version_changed",
    "managed_version_changed", "diff_json", "error",
]


def summary_row(row: dict, diff: dict, out: Path | None, error: str = "") -> dict:
    c = diff.get("counts", {}) if diff else {}
    pc = (diff or {}).get("pom_dependency_changes", {})
    d, m = pc.get("dependencies", {}), pc.get("managed", {})
    return {
        "project": row.get("project", ""),
        "classification": row.get("classification", ""),
        "group_id": row.get("dependency_group_id", ""),
        "artifact_id": row.get("dependency_artifact_id", ""),
        "old_version": row.get("old_version", ""),
        "new_version": row.get("new_version", ""),
        "status": (diff or {}).get("status", "ERROR"),
        "packaging": (diff or {}).get("packaging", ""),
        **{k: c.get(k, "") for k in ("removed", "changed", "added", "removed_classes",
                                    "added_classes", "old_methods", "new_methods")},
        "pom_deps_removed": len(d.get("removed", [])) if d else "",
        "pom_deps_added": len(d.get("added", [])) if d else "",
        "pom_deps_version_changed": len(d.get("version_changed", [])) if d else "",
        "managed_version_changed": len(m.get("version_changed", [])) if m else "",
        "diff_json": rel(out) if out else "",
        "error": error,
    }


def run_from_csv(csv_path: Path, limit: int) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    results = []
    done: dict[tuple, tuple[dict, Path | None, str]] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        if limit and i >= limit:
            break
        group = row.get("dependency_group_id") or ""
        artifact = row.get("dependency_artifact_id") or ""
        old_v = row.get("old_version") or ""
        new_v = row.get("new_version") or ""
        if not all([group, artifact, old_v, new_v]):
            continue
        key = (group, artifact, old_v, new_v)
        label = f"{group}:{artifact}:{old_v}->{new_v}"
        print(f"[{i+1}/{len(rows)}] {row.get('project', '')} {label}")
        if key not in done:
            out = OUT_DIR / f"{artifact}_{old_v}_to_{new_v}.json".replace("/", "_")
            try:
                diff = compare_coords(group, artifact, old_v, new_v)
                diff["projects"] = sorted({r.get("project", "") for r in rows
                                           if (r.get("dependency_group_id"), r.get("dependency_artifact_id"),
                                               r.get("old_version"), r.get("new_version")) == key})
                out.write_text(json.dumps(diff, indent=2), encoding="utf-8")
                done[key] = (diff, out, "")
                print(f"    {diff['status']} {diff.get('counts')}")
            except Exception as exc:  # network errors, missing artifacts, bad POMs
                print(f"    FAILED: {exc}")
                done[key] = ({}, None, str(exc))
        diff, out, err = done[key]
        results.append(summary_row(row, diff, out, err))
    summary = ROOT / "api_diff_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    from collections import Counter
    print("Status:", dict(Counter(r["status"] for r in results)))
    print("Status by class:", dict(Counter((r["classification"], r["status"]) for r in results)))
    print(f"Summary: {rel(summary)}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="API diff between two library JARs")
    parser.add_argument("--group")
    parser.add_argument("--artifact")
    parser.add_argument("--old")
    parser.add_argument("--new")
    parser.add_argument("--old-jar", type=Path)
    parser.add_argument("--new-jar", type=Path)
    parser.add_argument("--csv", type=Path, help="Dataset CSV with group/artifact/versions")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, help="Write single JSON result here")
    args = parser.parse_args()

    if args.csv:
        run_from_csv(args.csv, args.limit)
        return

    if args.old_jar and args.new_jar:
        result = compare_jars(args.old_jar, args.new_jar)
    elif args.group and args.artifact and args.old and args.new:
        result = compare_coords(args.group, args.artifact, args.old, args.new)
    else:
        parser.error("Provide --csv, or --old-jar/--new-jar, or Maven coordinates")
        return

    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print("Wrote", args.out)
    else:
        print(text)
    print("counts:", result.get("counts"))


if __name__ == "__main__":
    main()
