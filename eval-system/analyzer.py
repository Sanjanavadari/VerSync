"""
analyzer.py

VerSync dependency update risk prediction engine.

Given a repository, a dependency name, and an old/new version pair, it produces
a structured risk report (score, category, impacted files/functions,
reasoning, and a recommendation).

Graph-Model Integration:
Consumes a Dependency-Impact Graph (DIG) built by graph-model/build_dig.py.
Uses reverse BFS traversal (with cycle detection and depth capping) starting from
the target DEPENDENCY node back through APIs, direct function call sites, and
transitive internal callers to determine actual impacted files and blast radius.

Fallback Order:
  1. Explicit `--dig-path` argument if provided.
  2. `<repo_path>/dig.json` (cached graph in the repository).
  3. On-demand invocation of `graph-model/build_dig.py` (if javalang is available).
  4. Graceful fallback to deterministic mock report if no DIG can be loaded.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Risk category thresholds & configuration
# ---------------------------------------------------------------------------

SAFE_THRESHOLD = 0.4
RISKY_THRESHOLD = 0.7
MAX_BFS_DEPTH = 4  # Cap indirect call traversal to prevent noise & runaway loops

# Fallback canned pools (retained for mock fallback)
_MOCK_IMPACT_POOL: List[Tuple[str, str]] = [
    ("data_processing.py", "process_data()"),
    ("analytics.py", "calculate_metrics()"),
    ("serializers.py", "to_json()"),
    ("pipeline/transform.py", "apply_transform()"),
    ("pipeline/load.py", "bulk_insert()"),
    ("api/handlers.py", "handle_request()"),
    ("utils/validation.py", "validate_schema()"),
    ("models/dataset.py", "Dataset.__init__()"),
    ("reporting/export.py", "export_csv()"),
    ("core/config.py", "load_config()"),
]

_REASONING_BY_CATEGORY: Dict[str, List[str]] = {
    "SAFE": [
        "No breaking changes detected between the two versions in the changelog.",
        "Only patch-level fixes and internal refactors were found; public API is unchanged.",
        "Dependency is used in a narrow, well-isolated part of the codebase.",
    ],
    "RISKY": [
        "Uses deprecated APIs from {dependency} {old_version} that may be removed.",
        "Minor version bump introduces new default behavior in commonly used functions.",
        "Several call sites rely on behavior that may have changed across versions.",
    ],
    "CRITICAL": [
        "Major version bump removes or modifies APIs that are actively used across modules.",
        "Breaking change detected: function signature changed for a widely-called API.",
        "Dependency upgrade alters core behavior, risking runtime failures or regressions.",
    ],
}

_RECOMMENDATION_BY_CATEGORY: Dict[str, str] = {
    "SAFE": "Safe to upgrade. Run the standard test suite as a precaution.",
    "RISKY": "Review the flagged call sites before upgrading, and run targeted regression tests.",
    "CRITICAL": "Do not upgrade without a migration plan. Review all impacted files and "
    "consider pinning the current version until call sites are updated.",
}


def _categorize_risk(score: float, safe_threshold: float = SAFE_THRESHOLD,
                     risky_threshold: float = RISKY_THRESHOLD) -> str:
    """Map a numerical risk score to a risk category."""
    if score < safe_threshold:
        return "SAFE"
    if score < risky_threshold:
        return "RISKY"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# SemVer Analysis
# ---------------------------------------------------------------------------

def parse_semver(version_str: str) -> Tuple[int, int, int]:
    """Extract major, minor, patch ints from a version string."""
    numbers = re.findall(r"\d+", version_str)
    major = int(numbers[0]) if len(numbers) > 0 else 0
    minor = int(numbers[1]) if len(numbers) > 1 else 0
    patch = int(numbers[2]) if len(numbers) > 2 else 0
    return major, minor, patch


def classify_semver_bump(old_version: str, new_version: str) -> str:
    """Classify the update as 'MAJOR', 'MINOR', 'PATCH', or 'UNKNOWN'."""
    old_maj, old_min, old_pat = parse_semver(old_version)
    new_maj, new_min, new_pat = parse_semver(new_version)

    if new_maj > old_maj:
        return "MAJOR"
    if new_maj == old_maj and new_min > old_min:
        return "MINOR"
    if new_maj == old_maj and new_min == old_min and new_pat > old_pat:
        return "PATCH"
    if (new_maj, new_min, new_pat) == (old_maj, old_min, old_pat):
        return "SAME"
    return "MINOR"  # Default fallback for atypical version strings


# ---------------------------------------------------------------------------
# DIG Graph Loading & Fallback Resolution
# ---------------------------------------------------------------------------

# Directories ignored when checking whether a cached DIG is stale (same set the DIG builder skips).
_STALE_SKIP_DIRS = {".git", "target", "build", ".idea", ".vscode", "node_modules"}


def _dig_is_stale(dig_file: str, repo_path: str, slack_seconds: float = 2.0) -> bool:
    """
    True if any .java file or pom.xml under repo_path was modified after dig_file was written.

    A cached <repo>/dig.json is reused without rebuilding, so after the code changes it would
    silently describe an older version of the project. This detects edited or added files;
    it cannot detect deleted ones.
    """
    try:
        built = os.path.getmtime(dig_file)
    except OSError:
        return False
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in _STALE_SKIP_DIRS]
        for name in files:
            if name.endswith(".java") or name == "pom.xml":
                try:
                    if os.path.getmtime(os.path.join(root, name)) > built + slack_seconds:
                        return True
                except OSError:
                    continue
    return False


def _try_invoke_build_dig(repo_path: str) -> Optional[dict]:
    """Attempt to invoke graph-model/build_dig.py dynamically on repo_path."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate_scripts = [
        os.path.join(here, "..", "graph-model", "build_dig.py"),
        os.path.join(here, "graph-model", "build_dig.py"),
    ]
    script_path = None
    for p in candidate_scripts:
        if os.path.isfile(p):
            script_path = os.path.abspath(p)
            break

    if not script_path:
        return None

    # Check for javalang requirement before running
    try:
        import javalang  # noqa: F401
    except ImportError:
        print(
            "[VerSync] Notice: javalang is not installed. To auto-generate DIGs on demand, "
            "run `pip install -r graph-model/requirements.txt` (or install javalang).",
            file=sys.stderr,
        )
        return None

    temp_out = os.path.join(repo_path, "dig.json")
    try:
        print(f"[VerSync] Running build_dig.py on {repo_path}...", file=sys.stderr)
        cmd = [sys.executable, script_path, repo_path, temp_out]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if os.path.isfile(temp_out):
            with open(temp_out, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as exc:
        print(
            f"[VerSync] Notice: build_dig.py execution failed: {exc}",
            file=sys.stderr,
        )
    return None


def load_dig(repo_path: str, dig_path: Optional[str] = None) -> Tuple[Optional[dict], Optional[str]]:
    """
    Resolve and load the DIG using explicit fallback order:
      1. Explicit --dig-path argument (if provided)
      2. <repo_path>/dig.json (cached DIG inside the target repo)
      3. On-demand invocation of build_dig.py
    Returns: (dig_dict, source_description) or (None, None).
    """
    # 1. Explicit path
    if dig_path:
        norm_path = os.path.abspath(dig_path)
        if os.path.isfile(norm_path):
            with open(norm_path, "r", encoding="utf-8") as fh:
                return json.load(fh), f"explicit path ({dig_path})"
        print(f"[VerSync] Warning: specified --dig-path not found: {dig_path}", file=sys.stderr)

    # 2. Cached repo dig.json
    repo_dig = os.path.join(repo_path, "dig.json")
    if os.path.isfile(repo_dig):
        with open(repo_dig, "r", encoding="utf-8") as fh:
            return json.load(fh), f"cached repository DIG ({repo_dig})"

    # 3. Dynamic build_dig invocation
    built_dig = _try_invoke_build_dig(repo_path)
    if built_dig:
        return built_dig, f"on-demand build_dig.py ({repo_path})"

    return None, None


# ---------------------------------------------------------------------------
# Reverse Graph Traversal (Decision 2)
# ---------------------------------------------------------------------------

def normalize_dependency_query(query: str) -> Tuple[str, Optional[str]]:
    """
    Normalize dependency query string.
    Handles 'spring-web', 'org.springframework:spring-web', or 'spring-web:5.3.1'.
    Returns (cleaned_query, optional_version).
    """
    cleaned = query.strip()
    # Check if qualified with version, e.g. group:artifact:version or artifact:version
    parts = cleaned.split(":")
    if len(parts) == 3:
        # group:artifact:version
        return f"{parts[0]}:{parts[1]}", parts[2]
    if len(parts) == 2 and re.match(r"^\d", parts[1]):
        # artifact:version
        return parts[0], parts[1]
    return cleaned, None


def find_matching_dependency_nodes(nodes: List[dict], dep_query: str) -> List[str]:
    """
    Find matching DEPENDENCY node IDs in the DIG.
    Matches exact id, exact name, groupId:artifactId, artifactId alone, or substring.
    """
    target, _ = normalize_dependency_query(dep_query)
    target_lower = target.lower()

    matched_ids = []
    for node in nodes:
        if node.get("type") != "DEPENDENCY":
            continue
        node_id = node.get("id", "")
        name = node.get("name", "")
        # node_id typically looks like "dependency:org.springframework:spring-web"
        dep_val = node_id.replace("dependency:", "").strip().lower()
        name_lower = name.strip().lower()

        # Match exact coords or exact name
        if dep_val == target_lower or name_lower == target_lower:
            matched_ids.append(node_id)
            continue

        # Match artifactId portion (e.g. "spring-web" matches "org.springframework:spring-web")
        if ":" in dep_val and dep_val.split(":", 1)[1] == target_lower:
            matched_ids.append(node_id)
            continue
        if ":" in name_lower and name_lower.split(":", 1)[1] == target_lower:
            matched_ids.append(node_id)
            continue

        # Substring fallback match if specific enough
        if len(target_lower) >= 4 and (target_lower in dep_val or target_lower in name_lower):
            matched_ids.append(node_id)

    return list(dict.fromkeys(matched_ids))


def traverse_impacted_callsites(
    dig: dict,
    dep_node_ids: List[str],
    max_depth: int = MAX_BFS_DEPTH,
) -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, str]]:
    """
    Traverse the DIG backwards from the matched DEPENDENCY nodes.

    Guards implemented:
      - Cycle detection: maintains a visited set of FUNCTION nodes.
      - Depth cap: stops BFS at max_depth hops.

    Returns:
      - function_depths: dict of function_node_id -> min_hop_depth (1 = direct caller of API)
      - function_apis: dict of function_node_id -> list of API node ids called
      - function_to_file: dict of function_node_id -> relative file path
    """
    nodes_by_id = {n["id"]: n for n in dig.get("nodes", [])}
    edges = dig.get("edges", [])

    # Index reverse edges:
    # 1. API --PROVIDED_BY--> DEPENDENCY  ==>  api_by_dependency[dep_id] = [api_id, ...]
    api_by_dep: Dict[str, List[str]] = collections.defaultdict(list)
    # 2. FUNCTION --CALLS--> API / FUNCTION  ==>  callers_by_target[target_id] = [caller_func_id, ...]
    callers_by_target: Dict[str, List[str]] = collections.defaultdict(list)
    # 3. FILE --CONTAINS--> FUNCTION  ==>  file_by_function[func_id] = file_path
    file_by_function: Dict[str, str] = {}

    for edge in edges:
        s, t, etype = edge.get("source"), edge.get("target"), edge.get("type")
        if etype == "PROVIDED_BY":
            api_by_dep[t].append(s)
        elif etype == "CALLS":
            callers_by_target[t].append(s)
        elif etype == "CONTAINS":
            # source is FILE, target is FUNCTION
            if s.startswith("file:"):
                file_by_function[t] = s[len("file:"):]
            else:
                file_by_function[t] = s

    # Step 1: Collect all APIs provided by the matched dependencies
    target_apis: Set[str] = set()
    for dep_id in dep_node_ids:
        for api_id in api_by_dep.get(dep_id, []):
            target_apis.add(api_id)

    # Step 2: Find direct callers of these APIs (Depth 1)
    queue: collections.deque[Tuple[str, int]] = collections.deque()
    visited_functions: Set[str] = set()
    function_depths: Dict[str, int] = {}
    function_apis: Dict[str, List[str]] = collections.defaultdict(list)

    # sorted(): set iteration order of strings varies between Python processes (hash
    # randomization), which made the traversal order, and therefore tie order, non-reproducible.
    for api_id in sorted(target_apis):
        direct_callers = callers_by_target.get(api_id, [])
        for caller_func_id in direct_callers:
            function_apis[caller_func_id].append(api_id)
            if caller_func_id not in visited_functions:
                visited_functions.add(caller_func_id)
                function_depths[caller_func_id] = 1
                queue.append((caller_func_id, 1))

    # Step 3: BFS for transitive internal callers (Depth 2 .. max_depth)
    while queue:
        curr_func_id, depth = queue.popleft()
        if depth >= max_depth:
            continue

        for indirect_caller in callers_by_target.get(curr_func_id, []):
            if indirect_caller not in visited_functions:
                visited_functions.add(indirect_caller)
                function_depths[indirect_caller] = depth + 1
                queue.append((indirect_caller, depth + 1))

    return function_depths, dict(function_apis), file_by_function


def extract_graph_features(
    dig: dict,
    dependency: str,
    old_version: str,
    new_version: str,
) -> Dict[str, Any]:
    """
    Extract structured tabular features from the graph traversal for both the
    interim heuristic and downstream supervised baseline ML models.
    """
    matched_deps = find_matching_dependency_nodes(dig.get("nodes", []), dependency)
    func_depths, func_apis, func_to_file = traverse_impacted_callsites(dig, matched_deps)

    direct_calls = sum(1 for d in func_depths.values() if d == 1)
    indirect_calls = sum(1 for d in func_depths.values() if d > 1)
    affected_functions = len(func_depths)

    # Count distinct files
    affected_files_set = set()
    for func_id in func_depths:
        f = func_to_file.get(func_id)
        if not f and func_id.startswith("function:"):
            body = func_id[len("function:"):]
            f = body.split("::")[0] if "::" in body else "unknown"
        if f:
            affected_files_set.add(f)

    max_depth = max(func_depths.values()) if func_depths else 0
    bump_type = classify_semver_bump(old_version, new_version)

    return {
        "matched_dep_nodes": matched_deps,
        "direct_calls": direct_calls,
        "indirect_calls": indirect_calls,
        "affected_functions": affected_functions,
        "affected_files": len(affected_files_set),
        "affected_files_list": sorted(affected_files_set),
        "max_depth": max_depth,
        "bump_type": bump_type,
        "bump_major": 1 if bump_type == "MAJOR" else 0,
        "bump_minor": 1 if bump_type == "MINOR" else 0,
        "bump_patch": 1 if bump_type == "PATCH" else 0,
        "func_depths": func_depths,
        "func_apis": func_apis,
        "func_to_file": func_to_file,
    }


# ---------------------------------------------------------------------------
# Heuristic Scoring & Reasoning (Task 1 Display)
# ---------------------------------------------------------------------------

def _depth_to_probability(depth: int) -> float:
    """Decaying impact probability by call-chain hop distance."""
    if depth == 1:
        return 0.90
    if depth == 2:
        return 0.65
    if depth == 3:
        return 0.40
    return 0.20


def _compute_graph_heuristic_score(features: Dict[str, Any]) -> float:
    """Compute interim risk score combining graph blast radius and SemVer bump."""
    bump_mult = {
        "MAJOR": 1.0,
        "MINOR": 0.7,
        "PATCH": 0.35,
        "SAME": 0.1,
    }.get(features["bump_type"], 0.6)

    direct = features["direct_calls"]
    indirect = features["indirect_calls"]
    n_files = features["affected_files"]

    if direct == 0 and indirect == 0:
        # Dependency not called or not found
        if features["matched_dep_nodes"]:
            # Dependency exists in pom/graph, but no call sites reached
            return round(0.15 * bump_mult, 2)
        # Dependency not in graph at all
        return round(0.25 * bump_mult, 2)

    # Blast radius formula
    raw_blast = (direct * 0.18) + (indirect * 0.06) + (n_files * 0.12)
    score = min(0.98, max(0.10, raw_blast * bump_mult))
    return round(score, 2)


def _format_impacted_files_from_graph(
    func_depths: Dict[str, int],
    func_to_file: Dict[str, str],
) -> Dict[str, float]:
    """Format impacted files map: 'File -> Function' -> probability."""
    impacted: Dict[str, float] = {}

    for func_id, depth in func_depths.items():
        file_path = func_to_file.get(func_id)
        func_name = func_id

        if func_id.startswith("function:"):
            raw = func_id[len("function:"):]
            if "::" in raw:
                fpart, sep, mpart = raw.partition("::")
                file_path = file_path or fpart
                func_name = mpart
            else:
                func_name = raw

        file_display = os.path.basename(file_path) if file_path else "unknown_file"
        prob = _depth_to_probability(depth)
        key = f"{file_display} -> {func_name}()"
        impacted[key] = prob

    # Sort descending by probability
    return dict(sorted(impacted.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Fallback Mock Helpers (Preserved for graceful fallback)
# ---------------------------------------------------------------------------

def _deterministic_score(repo_path: str, dependency: str, old_version: str, new_version: str) -> float:
    key = f"{repo_path}|{dependency}|{old_version}|{new_version}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    int_val = int(digest[:8], 16)
    return round((int_val % 10_000) / 10_000, 2)


def _mock_impacted_files(score: float, seed_key: str) -> Dict[str, float]:
    rng = random.Random(seed_key)
    if score < SAFE_THRESHOLD:
        num_entries = rng.randint(0, 1)
    elif score < RISKY_THRESHOLD:
        num_entries = rng.randint(1, 3)
    else:
        num_entries = rng.randint(3, 5)

    chosen = rng.sample(_MOCK_IMPACT_POOL, k=min(num_entries, len(_MOCK_IMPACT_POOL)))
    impacted: Dict[str, float] = {}
    for file_name, func_name in chosen:
        jitter = rng.uniform(-0.1, 0.1)
        prob = max(0.05, min(0.99, score + jitter))
        key = f"{file_name} -> {func_name}"
        impacted[key] = round(prob, 2)
    return dict(sorted(impacted.items(), key=lambda kv: kv[1], reverse=True))


def _mock_reasoning(category: str, dependency: str, old_version: str, new_version: str, seed_key: str) -> str:
    rng = random.Random(seed_key + "|reasoning")
    template = rng.choice(_REASONING_BY_CATEGORY[category])
    return template.format(dependency=dependency, old_version=old_version, new_version=new_version)


def _mock_recommendation(category: str) -> str:
    return _RECOMMENDATION_BY_CATEGORY[category]


# ---------------------------------------------------------------------------
# Risk Triage & Impact Tree (Task 5)
# ---------------------------------------------------------------------------

# Evidence levels describe how much the graph tells us about this update.
EVIDENCE_CALL_SITES = "call-sites-found"
EVIDENCE_NO_CALL_SITES = "no-call-sites"
EVIDENCE_NOT_IN_GRAPH = "dependency-not-in-graph"
EVIDENCE_SIMULATED = "simulated"

_CONFIDENCE_BY_EVIDENCE = {
    EVIDENCE_CALL_SITES: "high",
    EVIDENCE_NO_CALL_SITES: "medium",
    EVIDENCE_NOT_IN_GRAPH: "low",
    EVIDENCE_SIMULATED: "none",
}


def _is_test_path(path: Optional[str]) -> bool:
    if not path:
        return False
    p = path.replace("\\", "/")
    base = p.rsplit("/", 1)[-1]
    return "/src/test/" in p or p.startswith("src/test/") or base.endswith(("Test.java", "Tests.java", "IT.java"))


def compute_blast_radius(features: Dict[str, Any]) -> Dict[str, int]:
    """Summarize the size of the affected region from extract_graph_features output."""
    func_to_file = features.get("func_to_file", {})
    in_tests = sum(1 for fid in features.get("func_depths", {}) if _is_test_path(func_to_file.get(fid)))
    return {
        "functions": features.get("affected_functions", 0),
        "files": features.get("affected_files", 0),
        "direct_call_sites": features.get("direct_calls", 0),
        "transitive_callers": features.get("indirect_calls", 0),
        "max_depth": features.get("max_depth", 0),
        "functions_in_test_files": in_tests,
    }


def compute_triage(
    features: Dict[str, Any],
    risk_score: float,
    dependency: str = "",
    old_version: str = "",
    new_version: str = "",
    safe_threshold: float = SAFE_THRESHOLD,
    risky_threshold: float = RISKY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Standardized SAFE / RISKY / CRITICAL triage with actionable next steps.

    The tier starts from the score thresholds (SAFE_THRESHOLD / RISKY_THRESHOLD) but SAFE is only
    allowed when the graph actually provides evidence:
      * dependency not found in the graph  -> at least RISKY (no evidence either way; on the
        57-sample DIG set 25 of the 31 such updates were breaking)
      * dependency found, no call sites, MAJOR bump -> at least RISKY
    CRITICAL is never assigned without call-site evidence because the score cannot reach 0.7
    without it.
    """
    score_category = _categorize_risk(risk_score, safe_threshold, risky_threshold)
    matched = bool(features.get("matched_dep_nodes"))
    n_funcs = features.get("affected_functions", 0)
    n_files = features.get("affected_files", 0)
    direct = features.get("direct_calls", 0)
    bump = features.get("bump_type", "")
    dep = dependency or "the dependency"
    rng = f"{old_version} -> {new_version}" if old_version or new_version else "the new version"

    tier = score_category
    if not matched:
        evidence = EVIDENCE_NOT_IN_GRAPH
        if tier == "SAFE":
            tier = "RISKY"
    elif n_funcs == 0:
        evidence = EVIDENCE_NO_CALL_SITES
        if tier == "SAFE" and bump == "MAJOR":
            tier = "RISKY"
    else:
        evidence = EVIDENCE_CALL_SITES

    if evidence == EVIDENCE_NOT_IN_GRAPH:
        summary = (
            f"'{dep}' was not found in the dependency graph, so its impact could not be measured "
            f"(only the {bump or 'SemVer'} bump is known). Treat this result as unverified."
        )
        steps = [
            "Run the full test suite before merging; static analysis has no evidence either way.",
            f"Read the release notes for {dep} ({rng}).",
            f"Check whether {dep} is a test, runtime or plugin dependency that the graph does not cover.",
        ]
    elif tier == "SAFE":
        if evidence == EVIDENCE_NO_CALL_SITES:
            summary = f"No project call sites reach {dep}; this {bump} update looks low risk."
        else:
            summary = f"Only a small, shallow set of call sites reaches {dep} ({direct} direct, {n_files} file(s))."
        summary += " This is not a guarantee: static analysis can miss reflection, configuration and runtime behaviour changes."
        steps = ["Run the existing test suite before merging.", "Merge only if it passes."]
    elif tier == "RISKY" and evidence == EVIDENCE_NO_CALL_SITES:
        summary = f"No call sites reach {dep}, but a {bump} bump can still remove or change APIs."
        steps = [
            f"Read the release notes for removed or renamed APIs ({rng}).",
            "Run the full test suite, including build and integration steps.",
        ]
    elif tier == "RISKY":
        summary = (
            f"Review the {direct} direct call site(s) across {n_files} affected file(s). "
            "Run targeted regression tests before merging."
        )
        steps = [
            "Review the listed call sites, starting with the direct callers (depth 1).",
            "Run targeted tests for the affected files.",
            f"Read the release notes for {dep} ({rng}).",
        ]
    else:  # CRITICAL
        summary = (
            f"Large blast radius: {n_funcs} function(s) across {n_files} file(s). "
            "Do not upgrade without checking breaking changes and updating call sites."
        )
        steps = [
            "Do not merge automatically.",
            "Review every listed file, starting with the direct callers (depth 1).",
            f"Read the release notes for {dep} ({rng}) and update affected call sites.",
            f"Run the full test suite and integration tests; consider pinning {dep} until reviewed.",
        ]

    return {
        "tier": tier,
        "score_category": score_category,
        "evidence": evidence,
        "confidence": _CONFIDENCE_BY_EVIDENCE[evidence],
        "summary": summary,
        "next_steps": steps,
    }


def _short_api(api_id: str) -> str:
    """'api:org.apache.mina.core.buffer.IoBuffer.allocate' -> 'IoBuffer.allocate'."""
    body = api_id[len("api:"):] if api_id.startswith("api:") else api_id
    parts = body.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else body


def build_impact_tree(
    dig: dict,
    dep_node_ids: List[str],
    dep_label: str = "",
    max_depth: int = MAX_BFS_DEPTH,
    max_children: int = 5,
    max_nodes: int = 40,
) -> Optional[Dict[str, Any]]:
    """
    Build a (truncated) tree of impacted functions rooted at the updated dependency.

    Depth-1 nodes are functions that call the dependency's APIs directly; each deeper level holds
    the callers of the level above. A function is attached to the first parent that reaches it, so
    the depths match traverse_impacted_callsites. Children are ordered by subtree size (largest
    blast radius first), then by path. To keep large repositories readable, at most `max_children`
    children are shown per node and `max_nodes` nodes overall; the rest are counted in "hidden".

    Returns None when there are no matched dependency nodes.
    """
    if not dep_node_ids:
        return None

    api_by_dep: Dict[str, List[str]] = collections.defaultdict(list)
    callers_by_target: Dict[str, List[str]] = collections.defaultdict(list)
    file_by_function: Dict[str, str] = {}
    for edge in dig.get("edges", []):
        s, t, etype = edge.get("source"), edge.get("target"), edge.get("type")
        if etype == "PROVIDED_BY":
            api_by_dep[t].append(s)
        elif etype == "CALLS":
            callers_by_target[t].append(s)
        elif etype == "CONTAINS":
            file_by_function[t] = s[len("file:"):] if isinstance(s, str) and s.startswith("file:") else s

    target_apis = sorted({a for d in dep_node_ids for a in api_by_dep.get(d, [])})
    called_apis: Dict[str, List[str]] = collections.defaultdict(list)
    for api_id in target_apis:
        for caller in callers_by_target.get(api_id, []):
            if api_id not in called_apis[caller]:
                called_apis[caller].append(api_id)

    depth: Dict[str, int] = {f: 1 for f in called_apis}
    children: Dict[str, List[str]] = collections.defaultdict(list)
    queue: collections.deque = collections.deque(sorted(called_apis))
    while queue:
        func = queue.popleft()
        if depth[func] >= max_depth:
            continue
        for caller in sorted(set(callers_by_target.get(func, []))):
            if caller not in depth:
                depth[caller] = depth[func] + 1
                children[func].append(caller)
                queue.append(caller)

    def file_of(func_id: str) -> str:
        f = file_by_function.get(func_id)
        if f:
            return f
        body = func_id[len("function:"):] if func_id.startswith("function:") else func_id
        return body.split("::")[0] if "::" in body else "unknown_file"

    def func_name(func_id: str) -> str:
        body = func_id[len("function:"):] if func_id.startswith("function:") else func_id
        return body.split("::", 1)[1] if "::" in body else body

    size_cache: Dict[str, int] = {}

    def size(func_id: str) -> int:
        if func_id not in size_cache:
            size_cache[func_id] = 1 + sum(size(c) for c in children.get(func_id, []))
        return size_cache[func_id]

    def ordered(func_ids: List[str]) -> List[str]:
        return sorted(func_ids, key=lambda f: (-size(f), file_of(f), func_name(f)))

    def make_node(func_id: str) -> Dict[str, Any]:
        path = file_of(func_id)
        d = depth[func_id]
        node: Dict[str, Any] = {
            "name": f"{os.path.basename(path)}::{func_name(func_id)}()",
            "file": path,
            "depth": d,
            "probability": _depth_to_probability(d),
            "is_test": _is_test_path(path),
            "children": [],
            "hidden": 0,
        }
        if d == 1:
            node["calls"] = sorted({_short_api(a) for a in called_apis[func_id]})[:3]
        return node

    level1 = ordered(list(called_apis))
    total = len(depth)
    root: Dict[str, Any] = {
        "name": dep_label or dep_node_ids[0].replace("dependency:", ""),
        "type": "dependency",
        "total_functions": total,
        "shown_functions": 0,
        "children": [],
        "hidden": 0,
    }

    budget = max_nodes
    work: collections.deque = collections.deque()
    work.append((None, root, level1))
    while work:
        _full, pruned, kids = work.popleft()
        shown = 0
        hidden_funcs = 0
        for kid in kids:
            if shown < max_children and budget > 0:
                budget -= 1
                shown += 1
                node = make_node(kid)
                pruned["children"].append(node)
                work.append((kid, node, ordered(children.get(kid, []))))
            else:
                hidden_funcs += size(kid)
        pruned["hidden"] = hidden_funcs
    root["shown_functions"] = total - _count_hidden(root)
    return root


def _count_hidden(node: Dict[str, Any]) -> int:
    return node.get("hidden", 0) + sum(_count_hidden(c) for c in node.get("children", []))


def render_impact_tree(tree: Optional[Dict[str, Any]]) -> str:
    """Render build_impact_tree output as plain-ASCII text (safe for any console)."""
    if not tree:
        return "(no impact tree: dependency not found in the graph)"

    lines: List[str] = [f"{tree['name']}"]

    def label(node: Dict[str, Any]) -> str:
        text = f"{node['name']}  [d{node['depth']}, {round(node['probability'] * 100)}%]"
        if node.get("is_test"):
            text += " [test]"
        if node.get("calls"):
            text += "  calls " + ", ".join(node["calls"])
        return text

    def walk(node: Dict[str, Any], prefix: str) -> None:
        kids = node.get("children", [])
        hidden = node.get("hidden", 0)
        for i, kid in enumerate(kids):
            last = i == len(kids) - 1 and not hidden
            lines.append(f"{prefix}{'`-- ' if last else '+-- '}{label(kid)}")
            walk(kid, prefix + ("    " if last else "|   "))
        if hidden:
            lines.append(f"{prefix}`-- ... {hidden} more function(s) not shown")

    walk(tree, "")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Public Interface
# ---------------------------------------------------------------------------

def analyze_update(
    repo_path: str,
    dependency: str,
    old_version: str,
    new_version: str,
    dig_path: Optional[str] = None,
) -> dict:
    """
    Perform a risk analysis for a dependency version upgrade.

    Args:
        repo_path: Path to the repository being analyzed.
        dependency: Name of the dependency being updated (e.g. "spring-web" or "org.springframework:spring-web").
        old_version: Current version string (e.g. "5.3.24").
        new_version: Target version string (e.g. "6.0.7").
        dig_path: Optional path to a precomputed DIG JSON file.

    Returns:
        Structured dictionary matching the expected schema:
        {
            "dependency": str,
            "old_version": str,
            "new_version": str,
            "risk_score": float,          # 0.0 - 1.0
            "risk_category": str,         # "SAFE" | "RISKY" | "CRITICAL"
            "impacted_files": {           # file/function -> impact probability
                "PetClinicConcurrencyTests.java -> testDuplicatePetName()": 0.90,
                ...
            },
            "reasoning": str,
            "recommendation": str,
            # added in Task 5 (existing keys unchanged; risk_category is now the triage tier):
            "analysis_mode": str,         # "graph" | "mock" (mock = simulated, no DIG found)
            "graph_source": str | None,   # where the DIG came from
            "graph_stale": bool,          # cached dig.json older than the sources (graph mode only)
            "score_category": str,        # tier from the score alone, before evidence rules
            "triage": dict,               # tier, evidence, confidence, summary, next_steps
            "blast_radius": dict | None,  # functions, files, direct/transitive, depth, test functions
            "impact_tree": dict | None,   # truncated caller tree; render with render_impact_tree()
        }
    """
    # 1. Attempt to load DIG via fallback order (Decision 1)
    dig_data, graph_source = load_dig(repo_path, dig_path)

    # An empty DIG (e.g. built on demand from a directory with no Java sources) carries no
    # information. Treat it as "no graph" so it cannot pass for a real analysis.
    if dig_data is not None and not dig_data.get("nodes"):
        print(f"[VerSync] Notice: the DIG from {graph_source} is empty (no Java sources found).", file=sys.stderr)
        dig_data = None

    graph_stale = False
    if dig_data is not None:
        print(f"[VerSync] Graph source: {graph_source}", file=sys.stderr)
        if graph_source and graph_source.startswith("cached repository DIG"):
            graph_stale = _dig_is_stale(os.path.join(repo_path, "dig.json"), repo_path)
            if graph_stale:
                print(
                    "[VerSync] Warning: sources changed after dig.json was built; the graph may be out of date. "
                    "Delete dig.json to rebuild it.",
                    file=sys.stderr,
                )

        # 2. Extract features and perform reverse BFS traversal (Decision 2)
        features = extract_graph_features(dig_data, dependency, old_version, new_version)
        risk_score = _compute_graph_heuristic_score(features)
        triage = compute_triage(features, risk_score, dependency, old_version, new_version)
        risk_category = triage["tier"]
        impacted_files = _format_impacted_files_from_graph(features["func_depths"], features["func_to_file"])

        # 3. Generate structured reasoning and recommendation
        bump_type = features["bump_type"]
        direct = features["direct_calls"]
        indirect = features["indirect_calls"]
        n_files = features["affected_files"]
        matched_deps = features["matched_dep_nodes"]

        if not matched_deps:
            reasoning = (
                f"Dependency '{dependency}' was not identified in the repository DIG. "
                f"Evaluated as a {bump_type} SemVer bump ({old_version} -> {new_version}) with no direct call sites detected."
            )
        elif direct == 0 and indirect == 0:
            dep_names = ", ".join(d.replace("dependency:", "") for d in matched_deps)
            reasoning = (
                f"Resolved dependency '{dep_names}' in the repository, but detected 0 active call sites. "
                f"Update is a {bump_type} SemVer bump with minimal localized surface risk."
            )
        else:
            dep_names = ", ".join(d.replace("dependency:", "") for d in matched_deps)
            reasoning = (
                f"Graph analysis detected {direct} direct API call site(s) and {indirect} transitive internal caller(s) "
                f"across {n_files} file(s) for '{dep_names}'. Combined with a {bump_type} SemVer upgrade "
                f"({old_version} -> {new_version}), the blast radius indicates {risk_category.lower()} update exposure."
            )

        recommendation = triage["summary"]

        return {
            "dependency": dependency,
            "old_version": old_version,
            "new_version": new_version,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "impacted_files": impacted_files,
            "reasoning": reasoning,
            "recommendation": recommendation,
            "analysis_mode": "graph",
            "graph_source": graph_source,
            "graph_stale": graph_stale,
            "score_category": triage["score_category"],
            "triage": triage,
            "blast_radius": compute_blast_radius(features) if matched_deps else None,  # None = not measured
            "impact_tree": build_impact_tree(dig_data, matched_deps, dep_label=dependency),
        }

    # 4. Fallback: If no graph available, run deterministic mock (with stderr notification)
    print(
        "[VerSync] Notice: Running mock risk analyzer (no DIG found via --dig-path or <repo>/dig.json).",
        file=sys.stderr,
    )
    seed_key = f"{repo_path}|{dependency}|{old_version}|{new_version}"
    risk_score = _deterministic_score(repo_path, dependency, old_version, new_version)
    impacted_files = _mock_impacted_files(risk_score, seed_key)
    risk_category = _categorize_risk(risk_score)
    reasoning = _mock_reasoning(risk_category, dependency, old_version, new_version, seed_key)
    recommendation = _mock_recommendation(risk_category)

    return {
        "dependency": dependency,
        "old_version": old_version,
        "new_version": new_version,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "impacted_files": impacted_files,
        "reasoning": reasoning,
        "recommendation": recommendation,
        "analysis_mode": "mock",
        "graph_source": None,
        "graph_stale": False,
        "score_category": risk_category,
        "triage": {
            "tier": risk_category,
            "score_category": risk_category,
            "evidence": EVIDENCE_SIMULATED,
            "confidence": _CONFIDENCE_BY_EVIDENCE[EVIDENCE_SIMULATED],
            "summary": "SIMULATED RESULT: no usable dependency graph was found, so this score and file list are not real analysis.",
            "next_steps": [
                "Build a DIG for this repository (graph-model) or pass --dig-path, then run again.",
                "Do not use this result to decide whether to merge.",
            ],
        },
        "blast_radius": None,
        "impact_tree": None,
    }


if __name__ == "__main__":
    # Test execution against sample DIG
    sample_dig = os.path.join(os.path.dirname(__file__), "..", "graph-model", "dig_sample_output.json")
    report = analyze_update(
        repo_path=".",
        dependency="spring-web",
        old_version="5.3.24",
        new_version="6.0.7",
        dig_path=sample_dig if os.path.exists(sample_dig) else None,
    )
    print(json.dumps(report, indent=2))
