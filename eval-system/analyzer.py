"""
analyzer.py

Mock/stub implementation of the dependency update risk prediction engine.

This module simulates what the real analysis pipeline will eventually do:
given a repo, a dependency name, and an old/new version pair, it produces
a structured risk report (score, category, impacted files/functions,
reasoning, and a recommendation).

Person 2 and Person 1 will later replace the internals of `analyze_update`
(and its helper functions) with:
  - real graph propagation logic over the dependency/call graph, and
  - an actual ML model for scoring risk.

The public function signature and the shape of the returned dictionary
are intended to stay stable so downstream consumers (CLI, API, tests)
don't need to change when the mock logic is swapped out.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Risk category thresholds
# ---------------------------------------------------------------------------

SAFE_THRESHOLD = 0.4
RISKY_THRESHOLD = 0.7

# A small pool of plausible-looking file/function targets used to fabricate
# "impacted files" data. In the real system this would come from static
# analysis / call-graph traversal instead of being sampled randomly.
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

# Canned reasoning strings, keyed loosely by risk category, used to make the
# mock output feel plausible without needing real analysis yet.
_REASONING_BY_CATEGORY: Dict[str, List[str]] = {
    "SAFE": [
        "No breaking changes detected between the two versions in the changelog.",
        "Only patch-level fixes and internal refactors were found; public API is unchanged.",
        "Dependency is used in a narrow, well-isolated part of the codebase.",
    ],
    "RISKY": [
        "Uses 3 deprecated APIs from {dependency} {old_version} that are marked for removal.",
        "Minor version bump introduces new default behavior in commonly used functions.",
        "Several call sites rely on undocumented behavior that may have changed.",
    ],
    "CRITICAL": [
        "Major version bump removes APIs that are actively used across multiple modules.",
        "Breaking change detected: function signature changed for a widely-called API.",
        "Dependency upgrade alters default data types returned by core functions, "
        "risking silent data corruption.",
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


def _deterministic_score(repo_path: str, dependency: str, old_version: str, new_version: str) -> float:
    """
    Produce a deterministic pseudo-random score in [0.0, 1.0] based on the
    inputs. Using a hash (rather than plain `random`) means repeated calls
    with the same arguments always return the same score, which is useful
    for tests and demos.
    """
    key = f"{repo_path}|{dependency}|{old_version}|{new_version}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    # Take the first 8 hex chars -> int -> normalize to [0, 1]
    int_val = int(digest[:8], 16)
    return round((int_val % 10_000) / 10_000, 2)


def _mock_impacted_files(score: float, seed_key: str) -> Dict[str, float]:
    """
    Fabricate a set of impacted file/function -> impact-probability pairs.
    Higher overall risk scores produce more impacted entries with higher
    individual probabilities, to keep the mock output internally consistent.
    """
    rng = random.Random(seed_key)  # deterministic per (repo, dep, versions)

    # Riskier updates "touch" more files.
    if score < SAFE_THRESHOLD:
        num_entries = rng.randint(0, 1)
    elif score < RISKY_THRESHOLD:
        num_entries = rng.randint(1, 3)
    else:
        num_entries = rng.randint(3, 5)

    chosen = rng.sample(_MOCK_IMPACT_POOL, k=min(num_entries, len(_MOCK_IMPACT_POOL)))

    impacted: Dict[str, float] = {}
    for file_name, func_name in chosen:
        # Individual impact probability loosely correlated with overall score,
        # with a bit of jitter so entries aren't identical.
        jitter = rng.uniform(-0.1, 0.1)
        prob = max(0.05, min(0.99, score + jitter))
        key = f"{file_name} -> {func_name}"
        impacted[key] = round(prob, 2)

    # Sort by descending impact probability for readability.
    return dict(sorted(impacted.items(), key=lambda kv: kv[1], reverse=True))


def _mock_reasoning(category: str, dependency: str, old_version: str, new_version: str, seed_key: str) -> str:
    rng = random.Random(seed_key + "|reasoning")
    template = rng.choice(_REASONING_BY_CATEGORY[category])
    return template.format(dependency=dependency, old_version=old_version, new_version=new_version)


def _mock_recommendation(category: str) -> str:
    return _RECOMMENDATION_BY_CATEGORY[category]


def analyze_update(repo_path: str, dependency: str, old_version: str, new_version: str) -> dict:
    """
    Simulate a risk analysis for a dependency version upgrade.

    Args:
        repo_path: Path to the repository being analyzed.
        dependency: Name of the dependency being updated (e.g. "pandas").
        old_version: Current version string (e.g. "2.1.0").
        new_version: Target version string (e.g. "2.2.0").

    Returns:
        A dictionary with the following shape:
        {
            "dependency": str,
            "old_version": str,
            "new_version": str,
            "risk_score": float,          # 0.0 - 1.0
            "risk_category": str,         # "SAFE" | "RISKY" | "CRITICAL"
            "impacted_files": {           # file/function -> impact probability
                "data_processing.py -> process_data()": 0.91,
                ...
            },
            "reasoning": str,
            "recommendation": str,
        }

    NOTE: This is a mock/stub implementation. Scores and impacted files are
    generated deterministically from the input arguments (via hashing), not
    from real static/graph analysis or an ML model. This will be replaced by
    Person 1 (ML model) and Person 2 (graph propagation logic).
    """
    seed_key = f"{repo_path}|{dependency}|{old_version}|{new_version}"

    # 1. Determine a deterministic mock score based on the inputs
    risk_score = _deterministic_score(repo_path, dependency, old_version, new_version)

    # 2. Build a list of impacted file/function paths
    impacted_files = _mock_impacted_files(risk_score, seed_key)

    # 3. Categorize the risk (SAFE, RISKY, CRITICAL)
    risk_category = _categorize_risk(risk_score)
    reasoning = _mock_reasoning(risk_category, dependency, old_version, new_version, seed_key)
    recommendation = _mock_recommendation(risk_category)

    # 4. Return a structured dictionary representing the analysis report

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
    # Quick manual smoke test / usage example.
    report = analyze_update(
        repo_path="/repos/example-service",
        dependency="pandas",
        old_version="2.1.4",
        new_version="2.2.0",
    )
    import json

    print(json.dumps(report, indent=2))