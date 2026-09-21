#!/usr/bin/env python3
"""
extract_call_graph.py

Walks a cloned Java repository, parses every .java file with javalang, and
builds a lightweight method-level call graph edge list distinguishing
internal calls (within the same project) from external API calls
(methods belonging to imported third-party / JDK libraries).

Usage:
    python extract_call_graph.py <path-to-cloned-repo> <output.json>

See README.md in this folder for setup instructions and known limitations.
"""

import argparse
import json
import os
import sys

import javalang
from javalang.tree import (
    ClassDeclaration,
    ConstructorDeclaration,
    EnumDeclaration,
    FieldDeclaration,
    FormalParameter,
    InterfaceDeclaration,
    LocalVariableDeclaration,
    MethodDeclaration,
    MethodInvocation,
)

# This javalang version predates Java `record` support, so record
# declarations are simply not parsed (see README limitations).
TYPE_DECL_KINDS = (ClassDeclaration, InterfaceDeclaration, EnumDeclaration)

METHOD_KINDS = (MethodDeclaration, ConstructorDeclaration)

# Common java.lang classes that are implicitly available without an import.
JAVA_LANG_BUILTINS = {
    "String", "Object", "System", "Math", "Integer", "Long", "Double", "Float",
    "Boolean", "Character", "Byte", "Short", "Void", "Number", "Thread",
    "Runnable", "Exception", "RuntimeException", "Error", "Throwable",
    "StringBuilder", "StringBuffer", "Iterable", "Comparable", "Class", "Enum",
    "Override", "SuppressWarnings",
}


def find_java_files(repo_path):
    java_files = []
    for root, dirs, files in os.walk(repo_path):
        # skip build/vcs directories to keep the pass fast and noise-free
        dirs[:] = [d for d in dirs if d not in (".git", "target", "build", ".idea", ".vscode")]
        for f in files:
            if f.endswith(".java"):
                java_files.append(os.path.join(root, f))
    return java_files


def type_name(type_node):
    """Best-effort simple name for a javalang Type node."""
    if type_node is None:
        return None
    return getattr(type_node, "name", None)


def filter_types(node, types):
    """
    Like Node.filter(), but accepts a tuple of types. javalang's own
    .filter() only matches a single exact type (it does not handle tuples
    passed to isinstance), so tuple-based filtering needs this helper.
    """
    for path, n in node:
        if isinstance(n, types):
            yield path, n


def iter_type_declarations(comp_unit):
    """Yield every class/interface/enum declaration in the file, including nested ones."""
    for _, node in filter_types(comp_unit, TYPE_DECL_KINDS):
        yield node


def build_project_symbol_table(parsed_units):
    """
    Pass 1: scan every successfully parsed file to learn which class names
    and methods belong to this project, so pass 2 can tell internal calls
    apart from external ones.
    """
    internal_classes = set()
    internal_methods_by_class = {}

    for _, comp_unit in parsed_units:
        for type_decl in iter_type_declarations(comp_unit):
            internal_classes.add(type_decl.name)
            methods = internal_methods_by_class.setdefault(type_decl.name, set())
            for _, m in type_decl.filter(MethodDeclaration):
                methods.add(m.name)

    return internal_classes, internal_methods_by_class


def build_import_maps(comp_unit):
    """
    Returns:
      type_to_fqn: simple class name -> fully qualified name (from concrete imports)
      static_method_map: statically-imported method name -> owning class FQN
      static_wildcard_classes: list of FQNs from `import static pkg.Class.*;`
    """
    type_to_fqn = {}
    static_method_map = {}
    static_wildcard_classes = []

    if comp_unit.imports:
        for imp in comp_unit.imports:
            path = imp.path
            if imp.static:
                if imp.wildcard:
                    static_wildcard_classes.append(path)
                else:
                    owner_fqn, _, member = path.rpartition(".")
                    static_method_map[member] = owner_fqn
            else:
                if imp.wildcard:
                    # Can't resolve simple names from a `import pkg.*;` without
                    # a classpath/index, so this is a documented limitation.
                    continue
                simple = path.rsplit(".", 1)[-1]
                type_to_fqn[simple] = path

    return type_to_fqn, static_method_map, static_wildcard_classes


def build_var_type_map(class_decl, method_node):
    """
    Best-effort variable-name -> declared-type-name map combining class
    fields with the method's own parameters and local variable declarations.
    Does not track enhanced-for-loop variables or catch-clause variables.
    """
    var_types = {}

    # class fields (so `field.someCall()` can be resolved)
    for _, field in class_decl.filter(FieldDeclaration):
        tname = type_name(field.type)
        for decl in field.declarators:
            var_types[decl.name] = tname

    # method parameters
    for param in getattr(method_node, "parameters", []) or []:
        if isinstance(param, FormalParameter):
            var_types[param.name] = type_name(param.type)

    # local variable declarations inside this method (excluding nested methods)
    for path, local in method_node.filter(LocalVariableDeclaration):
        # path[0] is method_node itself; only look at ancestors *below* it
        # to detect a nested/local/anonymous method scope.
        if any(isinstance(p, METHOD_KINDS) for p in path[1:]):
            continue  # belongs to a nested/local/anonymous method, not this one
        tname = type_name(local.type)
        for decl in local.declarators:
            var_types[decl.name] = tname

    return var_types


def resolve_call(qualifier, member, enclosing_class, var_types, type_to_fqn,
                  static_method_map, static_wildcard_classes,
                  internal_classes, internal_methods_by_class):
    """
    Decide whether a MethodInvocation is internal or external, and return
    the best-effort fully-qualified-ish name of what was called.
    Returns (calls_str, is_external).
    """
    # Case: bare call, e.g. helper() or assertEquals(a, b)
    if qualifier is None or qualifier == "":
        if member in static_method_map:
            return f"{static_method_map[member]}.{member}", True
        if member in internal_methods_by_class.get(enclosing_class, set()):
            return f"{enclosing_class}.{member}", False
        if static_wildcard_classes:
            # Ambiguous: could be local/inherited, or from a `import static X.*;`.
            # Best-effort guess: attribute to the first static-wildcard class.
            return f"{static_wildcard_classes[0]}.{member}", True
        # Default: assume it's a local/inherited method call.
        return f"{enclosing_class}.{member}", False

    if qualifier == "this":
        return f"{enclosing_class}.{member}", False

    if qualifier == "super":
        # We don't resolve the superclass type here; best-effort label.
        return f"{enclosing_class}(super).{member}", False

    # Qualifier is a variable/field name with a tracked declared type
    if qualifier in var_types and var_types[qualifier]:
        vtype = var_types[qualifier]
        if vtype in internal_classes:
            return f"{vtype}.{member}", False
        if vtype in type_to_fqn:
            return f"{type_to_fqn[vtype]}.{member}", True
        if vtype in JAVA_LANG_BUILTINS:
            return f"java.lang.{vtype}.{member}", True
        # Unresolved generic/type-param/wildcard-imported type
        return f"{vtype}.{member}", True

    # Qualifier looks like a class name used for a static call, e.g. Foo.bar()
    if qualifier in internal_classes:
        return f"{qualifier}.{member}", False

    if qualifier in type_to_fqn:
        return f"{type_to_fqn[qualifier]}.{member}", True

    if qualifier in JAVA_LANG_BUILTINS:
        return f"java.lang.{qualifier}.{member}", True

    # Unresolved qualifier (unknown field/var, or a class from a wildcard
    # import). Default to external — in practice most unresolved qualified
    # calls we see in this simplified pass are on JDK/library types
    # (List, Map, StringBuilder, etc.) rather than project classes.
    return f"{qualifier}.{member}", True


def extract_edges_for_file(rel_path, comp_unit, internal_classes, internal_methods_by_class):
    edges = []

    for type_decl in iter_type_declarations(comp_unit):
        class_name = type_decl.name
        type_to_fqn, static_method_map, static_wildcard_classes = build_import_maps(comp_unit)

        for _, method in filter_types(type_decl, METHOD_KINDS):
            method_name = getattr(method, "name", "<init>") or "<init>"
            var_types = build_var_type_map(type_decl, method)

            for path, invocation in method.filter(MethodInvocation):
                # path[0] is method itself; only ancestors below it can mean
                # this invocation actually belongs to a nested method scope.
                if any(isinstance(p, METHOD_KINDS) for p in path[1:]):
                    continue  # belongs to a nested/local/anonymous method

                calls_str, is_external = resolve_call(
                    invocation.qualifier,
                    invocation.member,
                    class_name,
                    var_types,
                    type_to_fqn,
                    static_method_map,
                    static_wildcard_classes,
                    internal_classes,
                    internal_methods_by_class,
                )

                edges.append({
                    "file": rel_path,
                    "function": method_name,
                    "calls": calls_str,
                    "external": is_external,
                })

    return edges


def main():
    parser = argparse.ArgumentParser(description="Extract a method-level call graph from a Java repository.")
    parser.add_argument("repo_path", help="Path to a locally cloned Java repository")
    parser.add_argument("output_json", help="Path to write the resulting edge-list JSON")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    if not os.path.isdir(repo_path):
        print(f"error: {repo_path} is not a directory", file=sys.stderr)
        sys.exit(1)

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
        except Exception as e:  # javalang raises JavaSyntaxError plus a few others
            failed_files.append((path, str(e)))

    print(f"parsed {len(parsed_units)} files successfully, {len(failed_files)} failed to parse")

    internal_classes, internal_methods_by_class = build_project_symbol_table(parsed_units)

    all_edges = []
    for path, comp_unit in parsed_units:
        rel_path = os.path.relpath(path, repo_path)
        edges = extract_edges_for_file(rel_path, comp_unit, internal_classes, internal_methods_by_class)
        all_edges.extend(edges)

    repo_name = os.path.basename(os.path.normpath(repo_path))
    output = {
        "repo": repo_name,
        "edges": all_edges,
    }

    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    external_count = sum(1 for e in all_edges if e["external"])
    internal_count = len(all_edges) - external_count
    print(f"wrote {len(all_edges)} edges ({internal_count} internal, {external_count} external) to {args.output_json}")

    if failed_files:
        print(f"\n{len(failed_files)} file(s) could not be parsed (skipped):", file=sys.stderr)
        for path, err in failed_files[:10]:
            print(f"  - {os.path.relpath(path, repo_path)}: {err}", file=sys.stderr)
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more", file=sys.stderr)


if __name__ == "__main__":
    main()
