#!/usr/bin/env python3
"""
build_dig.py

Builds a first version of the Dependency-Impact Graph (DIG): an explicit
node/edge graph connecting

    FILE --CONTAINS--> FUNCTION --CALLS--> API --PROVIDED_BY--> DEPENDENCY

Internal (project-to-project) calls are also included as
FUNCTION --CALLS--> FUNCTION edges, since knowing which of a project's own
functions call each other is useful for impact localization later, even
though it's not part of the external-dependency chain.

This script reuses extract_call_graph.py's parsing pipeline directly (same
javalang parsing, symbol table, and dependency resolution) rather than
re-implementing it, and just reshapes the resulting edge list into a graph.

Usage:
    python build_dig.py <path-to-cloned-repo> <output.json>

See README.md for the node/edge schema and known limitations.
"""

import argparse
import json
import os
import sys

import javalang

from extract_call_graph import (
    build_project_symbol_table,
    extract_edges_for_file,
    find_java_files,
    iter_type_declarations,
)
from dependency_resolver import load_declared_dependencies


def file_node_id(file_path):
    return f"file:{file_path}"


def function_node_id(file_path, function_name):
    # Scoped by file (not just class/function name) to avoid collisions
    # between same-named methods in different files/classes — an explicit
    # improvement over using a bare "function:<name>" id.
    return f"function:{file_path}::{function_name}"


def api_node_id(calls_str):
    return f"api:{calls_str}"


def dependency_node_id(dependency):
    return f"dependency:{dependency}"


def build_class_to_file_map(repo_path, parsed_units):
    """
    Map each internal class's simple name to the (first) file that declares
    it, so an internal call's target ("OtherClass.method") can be resolved
    to the *same* FUNCTION node that class's own declaration uses, instead
    of a disconnected, unscoped node. Classes with duplicate simple names
    across files keep whichever file was seen first (same collision
    limitation already documented for internal-call resolution).
    """
    class_to_file = {}
    for path, comp_unit in parsed_units:
        rel_path = os.path.relpath(path, repo_path)
        for type_decl in iter_type_declarations(comp_unit):
            class_to_file.setdefault(type_decl.name, rel_path)
    return class_to_file


def build_dig(repo_path):
    java_files = find_java_files(repo_path)
    print(f"found {len(java_files)} .java files under {repo_path}")

    parsed_units = []
    failed_files = []
    for path in java_files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
            comp_unit = javalang.parse.parse(source)
            parsed_units.append((path, comp_unit))
        except Exception as e:
            failed_files.append((path, str(e)))

    print(f"parsed {len(parsed_units)} files successfully, {len(failed_files)} failed to parse")

    internal_classes, internal_methods_by_class = build_project_symbol_table(parsed_units)
    declared_dependencies = load_declared_dependencies(repo_path)
    class_to_file = build_class_to_file_map(repo_path, parsed_units)
    dependency_cache = {}

    nodes = {}   # node_id -> node dict, de-duplicated
    edges = []
    seen_edges = set()  # (source, target, type) de-duplication

    def add_node(node_id, node_type, name):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "type": node_type, "name": name}

    def add_edge(source, target, edge_type):
        key = (source, target, edge_type)
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append({"source": source, "target": target, "type": edge_type})

    all_edges_for_reference = []  # kept for the summary print at the end

    for path, comp_unit in parsed_units:
        rel_path = os.path.relpath(path, repo_path)
        call_edges = extract_edges_for_file(
            rel_path, comp_unit, internal_classes, internal_methods_by_class,
            declared_dependencies, dependency_cache,
        )
        all_edges_for_reference.extend(call_edges)

        f_node = file_node_id(rel_path)
        add_node(f_node, "FILE", rel_path)

        for edge in call_edges:
            fn_node = function_node_id(rel_path, edge["function"])
            add_node(fn_node, "FUNCTION", edge["function"])
            add_edge(f_node, fn_node, "CONTAINS")

            if edge["external"]:
                a_node = api_node_id(edge["calls"])
                add_node(a_node, "API", edge["calls"])
                add_edge(fn_node, a_node, "CALLS")

                dependency = edge["dependency"]  # "jdk" / "unknown" / "group:artifact"
                d_node = dependency_node_id(dependency)
                add_node(d_node, "DEPENDENCY", dependency)
                add_edge(a_node, d_node, "PROVIDED_BY")
            else:
                # Internal call: try to resolve the target to the *same*
                # file-scoped function node its own declaration uses (so
                # multi-hop chains like process() -> helper() -> API stay
                # connected), falling back to an unscoped node (e.g. for
                # "EnclosingClass(super).method" or an unresolved owner
                # class) when that's not possible.
                owner, _, target_method = edge["calls"].rpartition(".")
                target_file = class_to_file.get(owner)
                if target_file:
                    target_node = function_node_id(target_file, target_method)
                    add_node(target_node, "FUNCTION", target_method)
                else:
                    target_node = f"function:{edge['calls']}"
                    add_node(target_node, "FUNCTION", edge["calls"])
                add_edge(fn_node, target_node, "CALLS")

    return {
        "repo": os.path.basename(os.path.normpath(repo_path)),
        "nodes": list(nodes.values()),
        "edges": edges,
    }, all_edges_for_reference, failed_files


def main():
    parser = argparse.ArgumentParser(
        description="Build a Dependency-Impact Graph (DIG) from a Java repository."
    )
    parser.add_argument("repo_path", help="Path to a locally cloned Java repository")
    parser.add_argument("output_json", help="Path to write the resulting DIG JSON")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    if not os.path.isdir(repo_path):
        print(f"error: {repo_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    dig, call_edges, failed_files = build_dig(repo_path)

    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(dig, fh, indent=2)

    node_counts = {}
    for node in dig["nodes"]:
        node_counts[node["type"]] = node_counts.get(node["type"], 0) + 1
    edge_counts = {}
    for edge in dig["edges"]:
        edge_counts[edge["type"]] = edge_counts.get(edge["type"], 0) + 1

    print(f"wrote {len(dig['nodes'])} nodes ({node_counts}) and "
          f"{len(dig['edges'])} edges ({edge_counts}) to {args.output_json}")

    if failed_files:
        print(f"\n{len(failed_files)} file(s) could not be parsed (skipped):", file=sys.stderr)
        for path, err in failed_files[:10]:
            print(f"  - {os.path.relpath(path, repo_path)}: {err}", file=sys.stderr)
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more", file=sys.stderr)


if __name__ == "__main__":
    main()
