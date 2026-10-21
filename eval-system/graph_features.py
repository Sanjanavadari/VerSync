"""
graph_features.py

Richer structural features from a Dependency-Impact Graph (DIG) for the
updated dependency, used by train_models.py as the "structural" feature arm.

Everything here is computed from the *pre-update* DIG and the dependency
coordinates only, so the features are outcome-independent (no label or
post-update information leaks in).

Reuses analyzer.find_matching_dependency_nodes / traverse_impacted_callsites so
"which functions are impacted" means exactly the same thing as in the existing
baseline graph features.

Feature groups:
  * Exposure of the dependency's API surface: number of distinct API nodes used,
    fan-in (callers per API), share of all third-party calls that go to it,
    PageRank share of the dependency node among declared dependencies.
  * Shape of the impact: depth distribution of impacted callers (1, 2, 3+ hops),
    mean depth, caller fan-out, how dependency-focused the direct callers are.
  * Spread: distinct files / Maven modules reached and the fraction of each,
    fraction of all functions reached, share of impacted files that are tests.
  * Whole-graph context: size, density, number of declared dependencies and
    the share of unresolved ("unknown") API calls (a graph-quality signal).
"""

from __future__ import annotations

import collections
import math
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzer import find_matching_dependency_nodes, traverse_impacted_callsites


STRUCTURAL_COLS = [
    "target_api_count_log",
    "api_fanin_mean_log",
    "api_fanin_max_log",
    "dep_call_share",
    "dep_pagerank_share",
    "depth_frac_1",
    "depth_frac_2",
    "depth_frac_3plus",
    "depth_mean",
    "caller_fanout_mean_log",
    "caller_dep_focus",
    "affected_modules",
    "frac_modules_reached",
    "frac_files_reached",
    "frac_functions_reached",
    "test_file_share",
    "graph_nodes_log",
    "graph_files_log",
    "graph_density_log10",
    "declared_deps",
    "unknown_api_share",
]

_NON_LIBRARY_DEPS = {"dependency:jdk", "dependency:unknown"}
_TEST_DIR_RE = re.compile(r"(^|/)(src/test|test|tests|it|src/it)(/|$)")


def _norm_path(p: str) -> str:
    # DIGs built on Windows store backslash-separated paths.
    return p.replace("\\", "/")


def _module_of(file_path: str) -> str:
    """Maven module = path prefix before the 'src' component ('.' for the root)."""
    parts = _norm_path(file_path).split("/")
    if "src" in parts:
        idx = parts.index("src")
        return "/".join(parts[:idx]) or "."
    return parts[0] if len(parts) > 1 else "."


def _pagerank(n: int, src: np.ndarray, dst: np.ndarray, damping: float = 0.85,
              iters: int = 100, tol: float = 1e-9) -> np.ndarray:
    """Plain power-iteration PageRank on a directed graph (dangling mass spread uniformly)."""
    if n == 0:
        return np.zeros(0)
    out_deg = np.bincount(src, minlength=n).astype(float)
    weights = 1.0 / out_deg[src]
    # M[dst, src] = 1/outdeg(src)
    M = sparse.csr_matrix((weights, (dst, src)), shape=(n, n))
    dangling = out_deg == 0
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r_new = damping * (M @ r + r[dangling].sum() / n) + (1.0 - damping) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def empty_structural_features() -> Dict[str, float]:
    return {c: 0.0 for c in STRUCTURAL_COLS}


def extract_structural_features(dig: Optional[dict], dependency: str) -> Dict[str, float]:
    """Compute STRUCTURAL_COLS for one (DIG, dependency) pair. Zeros if no DIG."""
    if dig is None:
        return empty_structural_features()

    nodes = dig.get("nodes", [])
    edges = dig.get("edges", [])
    node_type = {n["id"]: n.get("type") for n in nodes}

    files = [n["id"] for n in nodes if n.get("type") == "FILE"]
    functions = [n["id"] for n in nodes if n.get("type") == "FUNCTION"]
    dep_nodes = [n["id"] for n in nodes if n.get("type") == "DEPENDENCY"]
    library_deps = [d for d in dep_nodes if d not in _NON_LIBRARY_DEPS]

    api_dep: Dict[str, str] = {}
    calls_out: Dict[str, List[str]] = collections.defaultdict(list)
    callers_of: Dict[str, List[str]] = collections.defaultdict(list)
    for e in edges:
        s, t, et = e.get("source"), e.get("target"), e.get("type")
        if et == "PROVIDED_BY":
            api_dep[s] = t
        elif et == "CALLS":
            calls_out[s].append(t)
            callers_of[t].append(s)

    matched = find_matching_dependency_nodes(nodes, dependency)
    matched_set = set(matched)
    func_depths, func_apis, func_to_file = traverse_impacted_callsites(dig, matched)

    target_apis = [a for a, d in api_dep.items() if d in matched_set]
    fanins = [len(set(callers_of.get(a, []))) for a in target_apis]

    # Share of all calls into third-party (non-jdk, non-unknown) APIs that hit this dependency.
    lib_api_calls = 0
    target_api_calls = 0
    for a, d in api_dep.items():
        if d in _NON_LIBRARY_DEPS:
            continue
        k = len(callers_of.get(a, []))
        lib_api_calls += k
        if d in matched_set:
            target_api_calls += k
    dep_call_share = target_api_calls / lib_api_calls if lib_api_calls else 0.0

    # PageRank over FUNCTION->(FUNCTION|API) CALLS and API->DEPENDENCY PROVIDED_BY:
    # mass flowing into a dependency node measures how central it is to the codebase.
    idx = {nid: i for i, nid in enumerate(node_type)}
    pr_src, pr_dst = [], []
    for e in edges:
        if e.get("type") in ("CALLS", "PROVIDED_BY"):
            s, t = idx.get(e.get("source")), idx.get(e.get("target"))
            if s is not None and t is not None:
                pr_src.append(s)
                pr_dst.append(t)
    dep_pagerank_share = 0.0
    if pr_src and library_deps:
        pr = _pagerank(len(idx), np.array(pr_src), np.array(pr_dst))
        lib_mass = sum(pr[idx[d]] for d in library_deps)
        tgt_mass = sum(pr[idx[d]] for d in matched if d in idx and d in library_deps)
        dep_pagerank_share = tgt_mass / lib_mass if lib_mass > 0 else 0.0

    depths = list(func_depths.values())
    n_aff = len(depths)
    depth_frac_1 = sum(1 for d in depths if d == 1) / n_aff if n_aff else 0.0
    depth_frac_2 = sum(1 for d in depths if d == 2) / n_aff if n_aff else 0.0
    depth_frac_3 = sum(1 for d in depths if d >= 3) / n_aff if n_aff else 0.0
    depth_mean = float(np.mean(depths)) if depths else 0.0

    direct_callers = [f for f, d in func_depths.items() if d == 1]
    fanouts = [len(calls_out.get(f, [])) for f in direct_callers]
    focus = []
    for f in direct_callers:
        ext = [t for t in calls_out.get(f, []) if node_type.get(t) == "API"
               and api_dep.get(t) not in _NON_LIBRARY_DEPS]
        if ext:
            focus.append(sum(1 for t in ext if api_dep.get(t) in matched_set) / len(ext))

    aff_files = set()
    for f in func_depths:
        fp = func_to_file.get(f)
        if not fp and f.startswith("function:") and "::" in f:
            fp = f[len("function:"):].split("::")[0]
        if fp:
            aff_files.add(_norm_path(fp))
    all_file_paths = [_norm_path(f[len("file:"):]) for f in files]
    all_modules = {_module_of(p) for p in all_file_paths}
    aff_modules = {_module_of(p) for p in aff_files}
    test_share = (sum(1 for p in aff_files if _TEST_DIR_RE.search(p)) / len(aff_files)
                  if aff_files else 0.0)

    n_nodes, n_edges = len(nodes), len(edges)
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    api_nodes = [a for a in api_dep]
    unknown_share = (sum(1 for a in api_nodes if api_dep[a] == "dependency:unknown") / len(api_nodes)
                     if api_nodes else 0.0)

    return {
        "target_api_count_log": math.log1p(len(target_apis)),
        "api_fanin_mean_log": math.log1p(float(np.mean(fanins))) if fanins else 0.0,
        "api_fanin_max_log": math.log1p(max(fanins)) if fanins else 0.0,
        "dep_call_share": dep_call_share,
        "dep_pagerank_share": float(dep_pagerank_share),
        "depth_frac_1": depth_frac_1,
        "depth_frac_2": depth_frac_2,
        "depth_frac_3plus": depth_frac_3,
        "depth_mean": depth_mean,
        "caller_fanout_mean_log": math.log1p(float(np.mean(fanouts))) if fanouts else 0.0,
        "caller_dep_focus": float(np.mean(focus)) if focus else 0.0,
        "affected_modules": float(len(aff_modules)),
        "frac_modules_reached": len(aff_modules) / len(all_modules) if all_modules else 0.0,
        "frac_files_reached": len(aff_files) / len(files) if files else 0.0,
        "frac_functions_reached": n_aff / len(functions) if functions else 0.0,
        "test_file_share": test_share,
        "graph_nodes_log": math.log1p(n_nodes),
        "graph_files_log": math.log1p(len(files)),
        "graph_density_log10": math.log10(density) if density > 0 else 0.0,
        "declared_deps": float(len(library_deps)),
        "unknown_api_share": unknown_share,
    }
