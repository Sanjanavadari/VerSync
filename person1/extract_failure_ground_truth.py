"""
Task 4: Failure-localization ground truth from BUMP build logs (Paper 2).

Parses reproductionLogs/successfulReproductionLogs/*.log and extracts:
  - failed source files
  - failed methods / symbols (cannot find symbol, missing methods, etc.)
  - failed test classes / methods (Surefire)
  - error messages

Outputs:
  failure_ground_truth.csv
  failure_ground_truth.json
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUMP = ROOT.parent / "dataset" / "bump"
LOG_DIR = BUMP / "reproductionLogs" / "successfulReproductionLogs"
BENCH = BUMP / "data" / "benchmark"
OUT_CSV = ROOT / "failure_ground_truth.csv"
OUT_JSON = ROOT / "failure_ground_truth.json"

# /path/File.java:[line,col] message
COMP_ERR = re.compile(
    r"\[ERROR\]\s+(?P<file>/?(?:[^\s:\[]+\.java)):\[(?P<line>\d+),\d+\]\s*(?P<msg>.+)"
)
# symbol:   method foo(bar)
SYMBOL_METHOD = re.compile(
    r"symbol:\s+(?P<kind>method|variable|class|constructor)\s+(?P<name>[^\n]+)"
)
# location: class foo.Bar / variable x of type ...
LOCATION = re.compile(r"location:\s+(?P<loc>.+)")
# Surefire: FooTest.testBar:42 NullPointer or FooTest › testBar
SUREFIRE_FAIL = re.compile(
    r"(?P<cls>[\w.]+)\.(?P<method>\w+)\s*(?:Time elapsed:|:)\s*.*?(?:<<<\s*(?:FAILURE|ERROR)!|FAILED)",
    re.I,
)
SUREFIRE_FAIL2 = re.compile(
    r"^\[ERROR\]\s+(?P<cls>[\w.]+)\[(?P<method>\w+)\]",
    re.M,
)
TEST_RUN_LINE = re.compile(
    r"(?P<cls>[\w.$]+)\.(?P<method>\w+)\s+Time elapsed:.*?<<<\s*(?:FAILURE|ERROR)!",
    re.I,
)
PACKAGE_MISSING = re.compile(r"package\s+(?P<pkg>[\w.]+)\s+does not exist")
CANNOT_ACCESS = re.compile(r"cannot access\s+(?P<cls>[\w.]+)")


def read_log(path: Path) -> str:
    """BUMP logs are often stored as ustar members; strip the tar header."""
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    for marker in ("[INFO] Scanning", "[INFO] -----", "[ERROR] COMPILATION", "[ERROR] Failed"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:]
    # Drop leading non-text / ustar noise
    idx = text.find("[INFO]")
    if idx >= 0:
        return text[idx:]
    return text


def normalize_file(path: str) -> str:
    path = path.replace("\\", "/")
    # Strip docker/container prefixes like /quickfixj/
    parts = path.split("/")
    for i, p in enumerate(parts):
        if p in ("src", "main", "test", "java") or p.endswith(".java"):
            # keep from nearest project-ish root: look for src/
            if "src" in parts[i:]:
                si = parts.index("src", i)
                return "/".join(parts[si - 1:]) if si > 0 else "/".join(parts[si:])
            break
    if "src/" in path:
        return path[path.index("src/") :]
    return path.lstrip("/")


def parse_log(text: str) -> dict:
    files = []
    functions = []
    symbols = []
    tests = []
    messages = []

    for m in COMP_ERR.finditer(text):
        f = normalize_file(m.group("file"))
        msg = m.group("msg").strip()
        files.append(f)
        messages.append(f"{f}:{m.group('line')}: {msg}")
        # Look ahead a few lines for symbol/location
        start = m.end()
        window = text[start:start + 300]
        sm = SYMBOL_METHOD.search(window)
        loc = LOCATION.search(window)
        if sm:
            kind = sm.group("kind")
            name = sm.group("name").strip()
            symbols.append(f"{kind} {name}")
            if kind in ("method", "constructor"):
                functions.append(name.split("(")[0].strip())
        if loc:
            loc_s = loc.group("loc").strip()
            # "class quickfix.mina.ssl.SSLFilter" or "variable sslFilter of type ..."
            cm = re.search(r"class\s+([\w.$]+)", loc_s)
            if cm:
                functions.append(cm.group(1))

    for m in PACKAGE_MISSING.finditer(text):
        symbols.append(f"package {m.group('pkg')}")
        messages.append(f"package {m.group('pkg')} does not exist")

    for m in CANNOT_ACCESS.finditer(text):
        symbols.append(f"cannot access {m.group('cls')}")

    for rx in (TEST_RUN_LINE, SUREFIRE_FAIL, SUREFIRE_FAIL2):
        for m in rx.finditer(text):
            cls = m.group("cls")
            method = m.group("method")
            tests.append(f"{cls}#{method}")
            files.append(cls.replace(".", "/") + ".java")
            functions.append(f"{cls}.{method}")

    # Unique preserve order
    def uniq(items):
        seen = set()
        out = []
        for x in items:
            x = x.strip()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return {
        "failed_files": uniq(files),
        "failed_functions": uniq(functions),
        "failed_symbols": uniq(symbols),
        "failed_tests": uniq(tests),
        "error_messages": uniq(messages)[:40],
    }


def main() -> None:
    if not LOG_DIR.exists():
        raise SystemExit(f"BUMP log dir not found: {LOG_DIR}")

    meta = {}
    if BENCH.exists():
        for path in BENCH.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            dep = data.get("updatedDependency") or {}
            meta[path.stem] = {
                "project": data.get("project", ""),
                "pr_url": data.get("url", ""),
                "group_id": dep.get("dependencyGroupID", ""),
                "artifact_id": dep.get("dependencyArtifactID", ""),
                "old_version": dep.get("previousVersion", ""),
                "new_version": dep.get("newVersion", ""),
                "failure_category": data.get("failureCategory", ""),
            }

    rows = []
    logs = sorted(LOG_DIR.glob("*.log"))
    print(f"Parsing {len(logs)} BUMP reproduction logs...")
    empty = 0
    for i, log_path in enumerate(logs, 1):
        sha = log_path.stem
        text = read_log(log_path)
        parsed = parse_log(text)
        info = meta.get(sha, {})
        if not parsed["failed_files"] and not parsed["failed_tests"] and not parsed["failed_symbols"]:
            empty += 1
        row = {
            "breaking_commit": sha,
            "project": info.get("project", ""),
            "pr_url": info.get("pr_url", ""),
            "dependency_group_id": info.get("group_id", ""),
            "dependency_artifact_id": info.get("artifact_id", ""),
            "old_version": info.get("old_version", ""),
            "new_version": info.get("new_version", ""),
            "failure_category": info.get("failure_category", ""),
            "failed_files": "|".join(parsed["failed_files"]),
            "failed_functions": "|".join(parsed["failed_functions"]),
            "failed_symbols": "|".join(parsed["failed_symbols"]),
            "failed_tests": "|".join(parsed["failed_tests"]),
            "num_failed_files": len(parsed["failed_files"]),
            "num_failed_functions": len(parsed["failed_functions"]),
            "num_failed_tests": len(parsed["failed_tests"]),
            "error_messages": " || ".join(
                (m[:240] for m in parsed["error_messages"][:8])
            )[:4000],
            "log_path": str(log_path),
        }
        rows.append({**row, "_parsed": parsed})
        if i % 100 == 0:
            print(f"  {i}/{len(logs)}")

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "breaking_commit", "project", "pr_url",
            "dependency_group_id", "dependency_artifact_id",
            "old_version", "new_version", "failure_category",
            "failed_files", "failed_functions", "failed_symbols", "failed_tests",
            "num_failed_files", "num_failed_functions", "num_failed_tests",
            "error_messages", "log_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    json_rows = []
    for row in rows:
        parsed = row.pop("_parsed")
        json_rows.append({
            **{k: row[k] for k in row if k != "error_messages"},
            "failed_files": parsed["failed_files"],
            "failed_functions": parsed["failed_functions"],
            "failed_symbols": parsed["failed_symbols"],
            "failed_tests": parsed["failed_tests"],
            "error_messages": parsed["error_messages"],
        })
    OUT_JSON.write_text(json.dumps(json_rows, indent=2), encoding="utf-8")

    with_files = sum(1 for r in rows if int(r["num_failed_files"]) > 0)
    with_tests = sum(1 for r in rows if int(r["num_failed_tests"]) > 0)
    with_funcs = sum(1 for r in rows if int(r["num_failed_functions"]) > 0)
    print("\n========== SUMMARY ==========")
    print("Logs parsed:", len(rows))
    print("With failed files:", with_files)
    print("With failed functions/symbols:", with_funcs)
    print("With failed tests:", with_tests)
    print("No localization extracted:", empty)
    print("CSV:", OUT_CSV)
    print("JSON:", OUT_JSON)


if __name__ == "__main__":
    main()
