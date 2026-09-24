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


def _categorize_risk(score: float) -> str:
    """Map a numerical risk score to a risk category."""
    if score < SAFE_THRESHOLD:
        return "SAFE"
    if score < RISKY_THRESHOLD:
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

    for api_id in target_apis:
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
        }
    """
    # 1. Attempt to load DIG via fallback order (Decision 1)
    dig_data, graph_source = load_dig(repo_path, dig_path)

    if dig_data is not None:
        print(f"[VerSync] Graph source: {graph_source}", file=sys.stderr)

        # 2. Extract features and perform reverse BFS traversal (Decision 2)
        features = extract_graph_features(dig_data, dependency, old_version, new_version)
        risk_score = _compute_graph_heuristic_score(features)
        risk_category = _categorize_risk(risk_score)
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

        if risk_category == "SAFE":
            recommendation = "Safe to upgrade. Run automated test suite to verify behavior."
        elif risk_category == "RISKY":
            recommendation = (
                f"Review the {direct} direct call site(s) across {n_files} affected file(s). "
                "Run targeted regression tests before merging."
            )
        else:
            recommendation = (
                f"Major blast radius detected across {n_files} file(s). Do not upgrade without checking "
                "breaking changes in the release notes and updating call sites."
            )

        return {
            "dependency": dependency,
            "old_version": old_version,
            "new_version": new_version,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "impacted_files": impacted_files,
            "reasoning": reasoning,
            "recommendation": recommendation,
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