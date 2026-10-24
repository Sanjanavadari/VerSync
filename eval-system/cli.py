"""
cli.py

Command-line interface for VerSync's dependency update risk analysis.

Usage:
    python3 cli.py --repo-path /path/to/repo --dependency pandas \
        --old-version 2.1.4 --new-version 2.2.0 [--format text|json]

Parses CLI arguments, runs the analysis engine from analyzer.py (DIG-based; falls back to a
clearly-labelled simulated result if no DIG is available), and prints a structured report either as a premium plain-text
console report or as raw JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from analyzer import analyze_update, render_impact_tree


# ---------------------------------------------------------------------------
# Text report rendering
# ---------------------------------------------------------------------------

WIDTH = 70  # overall width of the ASCII report, in characters

# Status prefixes for each risk category. Kept as plain ASCII/emoji so the
# report renders consistently across terminals.
_STATUS_PREFIX = {
    "SAFE": "[OK]  ",
    "RISKY": "[!]   ",
    "CRITICAL": "[XX]  ",
}


def _status_prefix(risk_category: str) -> str:
    return _STATUS_PREFIX.get(risk_category, "[?]   ")


_EVIDENCE_TEXT = {
    "call-sites-found": "call sites found in the dependency graph",
    "no-call-sites": "dependency is in the graph, but no call sites reach it",
    "dependency-not-in-graph": "dependency NOT found in the graph (SemVer-only estimate)",
    "simulated": "SIMULATED - no usable dependency graph was available",
}


def _wrap(text: str, indent: str = "  ", width: int = WIDTH) -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _print_impacted_files_table(impacted_files: dict) -> None:
    """Print the 'POTENTIALLY AFFECTED FILES' table, sized to fit the content."""
    if not impacted_files:
        print("  No specific files/functions flagged for this update.")
        return

    file_col_header = "File -> Function"
    prob_col_header = "Probability"

    file_col_width = max(len(file_col_header), max(len(k) for k in impacted_files))
    prob_col_width = len(prob_col_header)

    border = f"  +-{'-' * file_col_width}-+-{'-' * prob_col_width}-+"
    header = f"  | {file_col_header.ljust(file_col_width)} | {prob_col_header.ljust(prob_col_width)} |"

    print(border)
    print(header)
    print(border)

    # Sort by descending probability so the riskiest entries appear first.
    for name, prob in sorted(impacted_files.items(), key=lambda kv: kv[1], reverse=True):
        pct = f"{round(prob * 100)}%"
        print(f"  | {name.ljust(file_col_width)} | {pct.ljust(prob_col_width)} |")

    print(border)


def print_text_report(report: dict, show_tree: bool = True) -> None:
    """
    Print a premium, human-readable console report for a risk analysis result.

    Expects `report` to contain the keys produced by analyzer.analyze_update,
    plus a "repo_path" key (added by main() before this is called). Keys added in
    Task 5 (triage, blast_radius, impact_tree, analysis_mode) are optional so older
    report dictionaries still render.
    """
    dependency = report["dependency"]
    old_version = report["old_version"]
    new_version = report["new_version"]
    repo_path = report.get("repo_path", "(unknown)")
    risk_score = report["risk_score"]
    risk_category = report["risk_category"]
    reasoning = report["reasoning"]
    recommendation = report["recommendation"]
    impacted_files = report["impacted_files"]
    triage = report.get("triage")
    blast = report.get("blast_radius")
    tree = report.get("impact_tree")
    is_mock = report.get("analysis_mode") == "mock"

    # 1. Header (with an unmissable banner when the result is simulated)
    print("=" * WIDTH)
    print("VERSYNC UPDATE RISK REPORT".center(WIDTH))
    print("=" * WIDTH)
    if is_mock:
        print("!" * WIDTH)
        print(_wrap("SIMULATED RESULT: no usable dependency graph (DIG) was found for this repository. "
                    "The score and files below are NOT real analysis. Pass --dig-path or build a DIG.", indent=""))
        print("!" * WIDTH)

    # 2. Metadata: dependency, old vs new version, repo path
    print(f"Dependency Update:  {dependency} ({old_version} -> {new_version})")
    print(f"Target Repository:  {repo_path}")
    if report.get("graph_source"):
        print(f"Graph source:       {report['graph_source'].split(' (')[0]}")
    if report.get("graph_stale"):
        print(_wrap("WARNING: sources changed after this graph was built, so it may be out of date. "
                    "Delete dig.json in the repository to rebuild it.", indent=" " * 20, width=WIDTH))
    print("-" * WIDTH)

    # 3. Risk score and triage classification
    prefix = "[SIM] " if is_mock else _status_prefix(risk_category)
    suffix = "  <- SIMULATED, not real analysis" if is_mock else ""
    print(f"OVERALL RISK:       {prefix}{risk_category} ({risk_score}){suffix}")
    if triage:
        evidence = _EVIDENCE_TEXT.get(triage["evidence"], triage["evidence"])
        print(f"Evidence:           {evidence} [confidence: {triage['confidence']}]")
        if triage.get("score_category") and triage["score_category"] != triage["tier"]:
            print(_wrap(f"(score alone: {triage['score_category']}; raised because the graph has no evidence of safety)", indent=" " * 20, width=WIDTH))
    if blast:
        print(
            f"Blast radius:       {blast['functions']} function(s) in {blast['files']} file(s) | "
            f"{blast['direct_call_sites']} direct | max depth {blast['max_depth']} | "
            f"{blast['functions_in_test_files']} in test files"
        )
    print("-" * WIDTH)

    # 4. Reasoning, recommendation and next steps
    print("Reasoning:")
    print(_wrap(reasoning))
    print("Actionable Recommendation:")
    print(_wrap(recommendation))
    if triage and triage.get("next_steps"):
        print("Next steps:")
        for i, step in enumerate(triage["next_steps"], start=1):
            print(textwrap.fill(step, width=WIDTH, initial_indent=f"  {i}. ", subsequent_indent="     "))
    print("-" * WIDTH)

    # 5. Impact tree (call chain from the dependency out to affected code)
    if show_tree and tree:
        print(f"IMPACT TREE ({tree['shown_functions']} of {tree['total_functions']} affected functions shown):")
        print("  d1 = calls the dependency directly, d2+ = calls a function above it")
        for line in render_impact_tree(tree).splitlines():
            print("  " + line)
        print("-" * WIDTH)

    # 6. Table of impacted files and their probabilities
    print("POTENTIALLY AFFECTED FILES:")
    _print_impacted_files_table(impacted_files)

    print("=" * WIDTH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # 1 & 2. Initialize argparse.ArgumentParser and add arguments
    parser = argparse.ArgumentParser(
        prog="versync",
        description="VerSync: predict the risk of a dependency version update.",
    )
    parser.add_argument("--repo-path", required=True, help="Path to the target repository.")
    parser.add_argument("--dependency", required=True, help="Name of the dependency being updated.")
    parser.add_argument("--old-version", required=True, help="Current version of the dependency.")
    parser.add_argument("--new-version", required=True, help="Target version of the dependency.")
    parser.add_argument(
        "--dig-path",
        default=None,
        help="Optional path to a precomputed Dependency-Impact Graph (DIG) JSON file.",
    )
    parser.add_argument(
        "--require-graph",
        action="store_true",
        help="Exit with code 2 instead of printing a simulated result when no DIG is found (useful in CI).",
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Do not print the impact tree in the text report.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for the report (default: text).",
    )

    # 3. Parse arguments
    args = parser.parse_args()

    # 4. Invoke analyze_update() from analyzer
    try:
        report = analyze_update(
            repo_path=args.repo_path,
            dependency=args.dependency,
            old_version=args.old_version,
            new_version=args.new_version,
            dig_path=args.dig_path,
        )
    except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
        print(f"Error: failed to analyze update ({exc})", file=sys.stderr)
        sys.exit(1)

    if args.require_graph and report.get("analysis_mode") != "graph":
        print(
            "Error: no dependency graph (DIG) found and --require-graph was set; refusing to print a simulated result.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Attach repo_path to the report so it's available for display/JSON output,
    # even though analyze_update() itself doesn't include it in its return value.
    report = {**report, "repo_path": args.repo_path}

    # 5. Switch output format based on --format flag
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report, show_tree=not args.no_tree)


if __name__ == "__main__":
    main()