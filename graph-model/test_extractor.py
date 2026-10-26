#!/usr/bin/env python3
"""
test_extractor.py

Lightweight, dependency-free validation for dependency_resolver.py,
extract_call_graph.py, and build_dig.py. Not a pytest suite (this project
otherwise has zero third-party test dependencies) — just plain functions
that assert and a tiny runner, in the same spirit as eval-system/evaluator.py.

Run with:
    python test_extractor.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

from dependency_resolver import resolve_dependency, load_declared_dependencies

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# dependency_resolver.py unit tests
# ---------------------------------------------------------------------------

def test_resolve_known_table_match():
    dep = resolve_dependency("com.fasterxml.jackson.databind.ObjectMapper.readValue", [])
    assert dep == "com.fasterxml.jackson.core:jackson-databind", dep


def test_resolve_jdk():
    dep = resolve_dependency("java.lang.String.trim", [])
    assert dep == "jdk", dep
    dep2 = resolve_dependency("javax.persistence.EntityManager.persist", [])
    # javax.persistence is in the known table, not treated as bare "jdk"
    assert dep2 == "javax.persistence:javax.persistence-api", dep2


def test_resolve_groupid_prefix_fallback():
    declared = [{"groupId": "com.example.customlib", "artifactId": "customlib-core", "version": "1.0"}]
    dep = resolve_dependency("com.example.customlib.Widget.render", declared)
    assert dep == "com.example.customlib:customlib-core", dep


def test_resolve_unknown_when_no_signal():
    dep = resolve_dependency("com.totallymadeup.lib.Thing.doStuff", [])
    assert dep == "unknown", dep


def test_resolve_unknown_when_no_package():
    # Unresolved bare qualifier calls (e.g. "bar.doWork" where bar's type
    # was never tracked) have no capitalized segment to anchor a package on.
    dep = resolve_dependency("bar.doWork", [])
    assert dep == "unknown", dep


def test_resolve_prefers_declared_sibling_artifact():
    # Several artifacts share io.dropwizard; the package names the right one.
    declared = [{"groupId": "io.dropwizard", "artifactId": a, "version": "2.0"}
                for a in ("dropwizard-core", "dropwizard-db", "dropwizard-client")]
    assert resolve_dependency("io.dropwizard.db.DataSourceFactory.build", declared) == "io.dropwizard:dropwizard-db"
    assert resolve_dependency("io.dropwizard.client.JerseyClientBuilder.build", declared) == \
        "io.dropwizard:dropwizard-client"
    assert resolve_dependency("io.dropwizard.setup.Environment.jersey", declared) == "io.dropwizard:dropwizard-core"


def test_resolve_sibling_overrides_known_table_only_when_more_specific():
    declared = [{"groupId": "org.junit.jupiter", "artifactId": a, "version": "5.9"}
                for a in ("junit-jupiter", "junit-jupiter-api", "junit-jupiter-params")]
    assert resolve_dependency("org.junit.jupiter.api.Assertions.assertEquals", declared) == \
        "org.junit.jupiter:junit-jupiter-api"
    assert resolve_dependency("org.junit.jupiter.params.provider.Arguments.of", declared) == \
        "org.junit.jupiter:junit-jupiter-params"


def test_resolve_log4j_api_vs_core():
    declared = [{"groupId": "org.apache.logging.log4j", "artifactId": "log4j-core", "version": "2.20"}]
    assert resolve_dependency("org.apache.logging.log4j.LogManager.getLogger", declared) == \
        "org.apache.logging.log4j:log4j-api"
    assert resolve_dependency("org.apache.logging.log4j.core.config.Configurator.initialize", declared) == \
        "org.apache.logging.log4j:log4j-core"


# ---------------------------------------------------------------------------
# pom.xml parsing
# ---------------------------------------------------------------------------

POM_XML = """<project>
  <groupId>com.example</groupId>
  <artifactId>demo</artifactId>
  <version>1.0.0</version>
  <properties>
    <jackson.version>2.15.2</jackson.version>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.apache.commons</groupId>
      <artifactId>commons-lang3</artifactId>
      <version>3.12.0</version>
    </dependency>
    <dependency>
      <groupId>com.fasterxml.jackson.core</groupId>
      <artifactId>jackson-databind</artifactId>
      <version>${jackson.version}</version>
    </dependency>
  </dependencies>
</project>
"""

FOO_JAVA = """package com.example;

import org.apache.commons.lang3.StringUtils;
import com.fasterxml.jackson.databind.ObjectMapper;

public class Foo {
    public String process(String input) throws Exception {
        String trimmed = StringUtils.trim(input);
        ObjectMapper mapper = new ObjectMapper();
        String result = mapper.writeValueAsString(trimmed);
        return helper(result);
    }

    private String helper(String s) {
        return s.toUpperCase();
    }
}
"""


def _make_sample_repo(tmp_dir):
    src_dir = os.path.join(tmp_dir, "src", "main", "java", "com", "example")
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "pom.xml"), "w") as fh:
        fh.write(POM_XML)
    with open(os.path.join(src_dir, "Foo.java"), "w") as fh:
        fh.write(FOO_JAVA)


def test_pom_property_placeholder_resolves():
    tmp_dir = tempfile.mkdtemp()
    try:
        _make_sample_repo(tmp_dir)
        deps = load_declared_dependencies(tmp_dir)
        by_artifact = {d["artifactId"]: d for d in deps}
        assert "jackson-databind" in by_artifact
        assert by_artifact["jackson-databind"]["version"] == "2.15.2", by_artifact["jackson-databind"]
        assert by_artifact["commons-lang3"]["groupId"] == "org.apache.commons"
    finally:
        shutil.rmtree(tmp_dir)


# ---------------------------------------------------------------------------
# End-to-end: extract_call_graph.py and build_dig.py against a real sample repo
# ---------------------------------------------------------------------------

def test_extract_call_graph_end_to_end():
    tmp_dir = tempfile.mkdtemp()
    try:
        _make_sample_repo(tmp_dir)
        out_path = os.path.join(tmp_dir, "out.json")
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, "extract_call_graph.py"), tmp_dir, out_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        data = json.load(open(out_path))
        edges = {(e["function"], e["calls"]): e for e in data["edges"]}

        trim_edge = edges[("process", "org.apache.commons.lang3.StringUtils.trim")]
        assert trim_edge["external"] is True
        assert trim_edge["dependency"] == "org.apache.commons:commons-lang3", trim_edge

        jackson_edge = edges[("process", "com.fasterxml.jackson.databind.ObjectMapper.writeValueAsString")]
        assert jackson_edge["external"] is True
        assert jackson_edge["dependency"] == "com.fasterxml.jackson.core:jackson-databind", jackson_edge

        internal_edge = edges[("process", "Foo.helper")]
        assert internal_edge["external"] is False
        assert internal_edge["dependency"] is None, internal_edge

        jdk_edge = edges[("helper", "java.lang.String.toUpperCase")]
        assert jdk_edge["external"] is True
        assert jdk_edge["dependency"] == "jdk", jdk_edge
    finally:
        shutil.rmtree(tmp_dir)


PARAM_TEST_JAVA = """package com.example;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.extension.ExtendWith;
import com.fasterxml.jackson.databind.ObjectMapper;

@ExtendWith(Foo.class)
public class FooTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {"a", "b"})
    void roundTrip(String s) {
        new Foo().process(s);
    }
}
"""


def test_constructor_and_annotation_uses_are_external_edges():
    tmp_dir = tempfile.mkdtemp()
    try:
        _make_sample_repo(tmp_dir)
        test_dir = os.path.join(tmp_dir, "src", "test", "java", "com", "example")
        os.makedirs(test_dir)
        with open(os.path.join(test_dir, "FooTest.java"), "w") as fh:
            fh.write(PARAM_TEST_JAVA)
        out_path = os.path.join(tmp_dir, "out.json")
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, "extract_call_graph.py"), tmp_dir, out_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        edges = {(e["function"], e["calls"]): e for e in json.load(open(out_path))["edges"]}

        param = edges[("roundTrip", "org.junit.jupiter.params.ParameterizedTest.<annotation>")]
        assert param["dependency"] == "org.junit.jupiter:junit-jupiter-params", param
        assert ("roundTrip", "org.junit.jupiter.params.provider.ValueSource.<annotation>") in edges
        # Class and field level uses go to the "<class>" pseudo-function.
        assert edges[("<class>", "org.junit.jupiter.api.extension.ExtendWith.<annotation>")]["external"] is True
        mapper_new = edges[("<class>", "com.fasterxml.jackson.databind.ObjectMapper.<init>")]
        assert mapper_new["dependency"] == "com.fasterxml.jackson.core:jackson-databind", mapper_new
        # `new Foo()` is a project class, not an external API use.
        assert not any(c.startswith("Foo.<init>") or c.endswith(".Foo.<init>") for _, c in edges)
    finally:
        shutil.rmtree(tmp_dir)


def test_build_dig_chain_connectivity():
    tmp_dir = tempfile.mkdtemp()
    try:
        _make_sample_repo(tmp_dir)
        out_path = os.path.join(tmp_dir, "dig.json")
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, "build_dig.py"), tmp_dir, out_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        dig = json.load(open(out_path))
        nodes_by_id = {n["id"]: n for n in dig["nodes"]}
        by_type = {}
        for n in dig["nodes"]:
            by_type.setdefault(n["type"], []).append(n)
        assert len(by_type.get("FILE", [])) == 1
        assert len(by_type.get("FUNCTION", [])) == 2  # process, helper
        assert len(by_type.get("API", [])) == 4        # trim, ObjectMapper.<init>, writeValueAsString, toUpperCase
        assert len(by_type.get("DEPENDENCY", [])) == 3  # commons-lang3, jackson-databind, jdk

        # FILE --CONTAINS--> FUNCTION(process) --CALLS--> API(trim) --PROVIDED_BY--> DEPENDENCY(commons-lang3)
        file_node = next(n for n in dig["nodes"] if n["type"] == "FILE")
        process_node = next(n for n in dig["nodes"] if n["type"] == "FUNCTION" and n["name"] == "process")
        trim_api = next(n for n in dig["nodes"] if n["type"] == "API" and n["name"].endswith("StringUtils.trim"))
        commons_dep = next(n for n in dig["nodes"] if n["type"] == "DEPENDENCY" and "commons-lang3" in n["name"])

        def has_edge(source_id, target_id, edge_type):
            return any(e["source"] == source_id and e["target"] == target_id and e["type"] == edge_type
                       for e in dig["edges"])

        assert has_edge(file_node["id"], process_node["id"], "CONTAINS")
        assert has_edge(process_node["id"], trim_api["id"], "CALLS")
        assert has_edge(trim_api["id"], commons_dep["id"], "PROVIDED_BY")

        # Connectivity fix: process()'s internal call to helper() must land on
        # the SAME function node that helper()'s own CONTAINS/CALLS edges use
        # (not a disconnected, unscoped duplicate), so the chain
        # process -> helper -> String.toUpperCase -> jdk is walkable in one hop.
        helper_node = next(n for n in dig["nodes"] if n["type"] == "FUNCTION" and n["name"] == "helper")
        assert has_edge(process_node["id"], helper_node["id"], "CALLS")
        assert has_edge(file_node["id"], helper_node["id"], "CONTAINS")
        toupper_api = next(n for n in dig["nodes"] if n["type"] == "API" and n["name"].endswith("String.toUpperCase"))
        assert has_edge(helper_node["id"], toupper_api["id"], "CALLS")
    finally:
        shutil.rmtree(tmp_dir)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
              if name.startswith("test_") and callable(fn)]

    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
