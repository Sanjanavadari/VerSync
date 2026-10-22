"""
Task 2: API-diff extractor — compare public methods in old vs new library JARs.

Downloads JARs from Maven Central (or accepts local paths), runs `javap -public`,
and reports removed / changed / added method signatures.

Usage:
  python api_diff_extractor.py --group org.slf4j --artifact slf4j-api \\
      --old 1.7.36 --new 2.0.3

  python api_diff_extractor.py --csv final_verified_dataset.csv --limit 10

  python api_diff_extractor.py --old-jar path/old.jar --new-jar path/new.jar
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "jar_cache"
OUT_DIR = ROOT / "api_diffs"
MAVEN = "https://repo1.maven.org/maven2"

METHOD_RE = re.compile(
    r"^\s*(?:public|protected)\s+(?:static\s+|final\s+|native\s+|synchronized\s+)*"
    r"(?P<ret>[\w.<>,?\[\]\s]+?)\s+(?P<name>[\w$]+)\s*\((?P<params>[^)]*)\)"
)
CTOR_RE = re.compile(
    r"^\s*(?:public|protected)\s+(?P<name>[\w$]+)\s*\((?P<params>[^)]*)\)"
)


def maven_url(group: str, artifact: str, version: str) -> str:
    gpath = group.replace(".", "/")
    return f"{MAVEN}/{gpath}/{artifact}/{version}/{artifact}-{version}.jar"


def download_jar(group: str, artifact: str, version: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{group}_{artifact}_{version}.jar"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = maven_url(group, artifact, version)
    print(f"  Downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"JAR not found at {url}: {exc}") from exc
    return dest


def list_class_names(jar_path: Path) -> list[str]:
    names = []
    with zipfile.ZipFile(jar_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith(".class") and not name.endswith("module-info.class"):
                if "$" in Path(name).name:
                    # Keep inner classes; javap can still read them
                    pass
                names.append(name[:-6].replace("/", "."))
    return names


def parse_javap_batch(javap_out: str) -> dict[str, str]:
    """Parse `javap` output that may contain multiple class dumps."""
    methods: dict[str, str] = {}
    current_class = ""
    simple = ""
    for line in javap_out.splitlines():
        header = re.match(r"^(?:public\s+)?(?:final\s+)?(?:abstract\s+)?"
                          r"(?:class|interface|enum)\s+([\w.$]+)", line)
        if header:
            current_class = header.group(1)
            simple = current_class.rsplit(".", 1)[-1]
            continue
        if not current_class:
            continue
        line = line.rstrip()
        m = METHOD_RE.match(line)
        if m:
            name = m.group("name")
            params = re.sub(r"\s+", " ", m.group("params").strip())
            ret = re.sub(r"\s+", " ", m.group("ret").strip())
            key = f"{current_class}#{name}({params})"
            methods[key] = f"{ret} {name}({params})"
            continue
        c = CTOR_RE.match(line)
        if c and c.group("name") == simple:
            params = re.sub(r"\s+", " ", c.group("params").strip())
            key = f"{current_class}#{simple}({params})"
            methods[key] = f"{simple}({params})"
    return methods


def extract_api(jar_path: Path, batch_size: int = 40) -> dict[str, str]:
    classes = [
        cls for cls in list_class_names(jar_path)
        if not re.search(r"\$\d+$", cls)
    ]
    api: dict[str, str] = {}
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
            api.update(parse_javap_batch(proc.stdout or ""))
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"    javap batch error at {i}: {exc}")
    return api


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
    old_api = extract_api(old_jar)
    print(f"  Extracting API from {new_jar.name} ...")
    new_api = extract_api(new_jar)
    result = diff_apis(old_api, new_api)
    result["old_jar"] = str(old_jar)
    result["new_jar"] = str(new_jar)
    return result


def compare_coords(group: str, artifact: str, old_v: str, new_v: str) -> dict:
    old_jar = download_jar(group, artifact, old_v)
    new_jar = download_jar(group, artifact, new_v)
    result = compare_jars(old_jar, new_jar)
    result.update({
        "group_id": group,
        "artifact_id": artifact,
        "old_version": old_v,
        "new_version": new_v,
    })
    return result


def run_from_csv(csv_path: Path, limit: int) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    results = []
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
        
        # BOMs are dependency-management artifacts, not necessarily API JARs.
        bom_artifact = (
            "bom" in artifact.lower()
            or artifact.lower().endswith("-dependencies")
        )
        if bom_artifact:
            results.append({
                "project": row.get("project", ""),
                "group_id": group,
                "artifact_id": artifact,
                "old_version": old_v,
                "new_version": new_v,
                "removed": "N/A",
                "changed": "N/A",
                "added": "N/A",
                "old_methods": "N/A",
                "new_methods": "N/A",
                "diff_json": "NOT_APPLICABLE: BOM/dependency-management artifact; no direct API JAR comparison",
            })
            print(f"  Skipping BOM artifact: {artifact}")
            continue


        # Skip BOMs / plugins that often have no useful public API surface
        if artifact.endswith("-bom") or "plugin" in artifact.lower():
            print(f"[{i+1}] skip BOM/plugin {group}:{artifact}")
            continue
        label = f"{group}:{artifact}:{old_v}->{new_v}"
        print(f"[{i+1}] {label}")
        try:
            diff = compare_coords(group, artifact, old_v, new_v)
            diff["project"] = row.get("project", "")
            out = OUT_DIR / f"{artifact}_{old_v}_to_{new_v}.json".replace("/", "_")
            out.write_text(json.dumps(diff, indent=2), encoding="utf-8")
            results.append({
                "project": row.get("project", ""),
                "group_id": group,
                "artifact_id": artifact,
                "old_version": old_v,
                "new_version": new_v,
                "removed": diff["counts"]["removed"],
                "changed": diff["counts"]["changed"],
                "added": diff["counts"]["added"],
                "old_methods": diff["counts"]["old_methods"],
                "new_methods": diff["counts"]["new_methods"],
                "diff_json": str(out),
            })
            print(f"    removed={diff['counts']['removed']} "
                  f"changed={diff['counts']['changed']} "
                  f"added={diff['counts']['added']}")
        except Exception as exc:
            print(f"    FAILED: {exc}")
            results.append({
                "project": row.get("project", ""),
                "group_id": group,
                "artifact_id": artifact,
                "old_version": old_v,
                "new_version": new_v,
                "removed": "",
                "changed": "",
                "added": "",
                "old_methods": "",
                "new_methods": "",
                "diff_json": f"ERROR: {exc}",
            })
    summary = ROOT / "api_diff_summary.csv"
    fields = [
        "project", "group_id", "artifact_id", "old_version", "new_version",
        "removed", "changed", "added", "old_methods", "new_methods", "diff_json",
    ]
    with summary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Summary: {summary}")
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
