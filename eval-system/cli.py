"""
cli.py

Command-line interface for VerSync's dependency update risk analysis.

Usage:
    python3 cli.py --repo-path /path/to/repo --dependency pandas \
        --old-version 2.1.4 --new-version 2.2.0 [--format text|json]

Parses CLI arguments, runs the (currently mock) analysis engine from
analyzer.py, and prints a structured report either as a premium plain-text
console report or as raw JSON.
"""

from __future__ import annotations

import argparse
import json
import sys

from analyzer import analyze_update


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


def print_text_report(report: dict) -> None:
    """
    Print a premium, human-readable console report for a risk analysis result.

    Expects `report` to contain the keys produced by analyzer.analyze_update,
    plus a "repo_path" key (added by main() before this is called).
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

    # 1. Header
    print("=" * WIDTH)
    print("VERSYNC UPDATE RISK REPORT".center(WIDTH))
    print("=" * WIDTH)

    # 2. Metadata: dependency, old vs new version, repo path
    print(f"Dependency Update:  {dependency} ({old_version} -> {new_version})")
    print(f"Target Repository:  {repo_path}")
    print("-" * WIDTH)

    # 3. Risk score and classification, with a status prefix depending on category
    prefix = _status_prefix(risk_category)
    print(f"OVERALL RISK:       {prefix}{risk_category} ({risk_score})")
    print("-" * WIDTH)

    # 4. Reasoning and recommendation
    print("Reasoning:")
    print(f"  {reasoning}")
    print("Actionable Recommendation:")
    print(f"  {recommendation}")
    print("-" * WIDTH)

    # 5. Table of impacted files and their probabilities
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
        )
    except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
        print(f"Error: failed to analyze update ({exc})", file=sys.stderr)
        sys.exit(1)

    # Attach repo_path to the report so it's available for display/JSON output,
    # even though analyze_update() itself doesn't include it in its return value.
    report = {**report, "repo_path": args.repo_path}

    # 5. Switch output format based on --format flag
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()