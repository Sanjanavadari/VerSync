"""
dependency_resolver.py

Best-effort mapping from an external Java API call (e.g.
"com.fasterxml.jackson.databind.ObjectMapper.readValue") to the Maven
dependency that provides it (e.g. "com.fasterxml.jackson.core:jackson-databind").

This is deliberately NOT full type resolution. It combines two cheap,
explainable signals:

  1. A curated table of well-known package-prefix -> "groupId:artifactId"
     mappings for commonly used libraries (JUnit, Jackson, Guava, Commons,
     Spring, etc.) — necessary because many libraries' package names don't
     follow their Maven coordinates (e.g. Guava's groupId is
     "com.google.guava" but its packages are "com.google.common.*").
  2. The project's own declared Maven dependencies (parsed from every
     pom.xml found in the repo): if a call's package starts with a
     declared dependency's groupId, that's a solid heuristic match for
     libraries that *do* follow the groupId-as-package-prefix convention
     (e.g. org.apache.commons:commons-lang3 -> org.apache.commons.lang3.*).

If neither signal produces a match, we do not guess — we return "unknown".
See README.md for the full list of known prefixes and limitations.
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

SKIP_DIRS = {".git", "target", "build", ".idea", ".vscode", "node_modules"}

# Special, documented non-Maven dependency values.
JDK = "jdk"
UNKNOWN = "unknown"

# Curated package-prefix -> "groupId:artifactId" table for common libraries
# whose packages don't reliably derive from their Maven coordinates. Longest
# matching prefix wins, so more specific entries should stay specific rather
# than collapsed into a broader prefix (e.g. Jackson's submodules).
KNOWN_PACKAGE_PREFIXES: Dict[str, str] = {
    "org.junit.jupiter": "org.junit.jupiter:junit-jupiter-api",
    "org.junit": "junit:junit",
    "org.hamcrest": "org.hamcrest:hamcrest",
    "org.mockito": "org.mockito:mockito-core",
    "org.assertj": "org.assertj:assertj-core",
    "com.fasterxml.jackson.databind": "com.fasterxml.jackson.core:jackson-databind",
    "com.fasterxml.jackson.core": "com.fasterxml.jackson.core:jackson-core",
    "com.fasterxml.jackson.annotation": "com.fasterxml.jackson.core:jackson-annotations",
    "com.google.gson": "com.google.code.gson:gson",
    "com.google.common": "com.google.guava:guava",
    "com.google.protobuf": "com.google.protobuf:protobuf-java",
    "com.google.inject": "com.google.inject:guice",
    "org.apache.commons.lang3": "org.apache.commons:commons-lang3",
    "org.apache.commons.lang": "commons-lang:commons-lang",
    "org.apache.commons.io": "commons-io:commons-io",
    "org.apache.commons.collections4": "org.apache.commons:commons-collections4",
    "org.apache.commons.codec": "commons-codec:commons-codec",
    "org.apache.commons.text": "org.apache.commons:commons-text",
    "org.apache.commons.csv": "org.apache.commons:commons-csv",
    "org.slf4j": "org.slf4j:slf4j-api",
    "org.apache.logging.log4j": "org.apache.logging.log4j:log4j-core",
    "ch.qos.logback": "ch.qos.logback:logback-classic",
    "org.springframework.boot": "org.springframework.boot:spring-boot",
    "org.springframework.web": "org.springframework:spring-web",
    "org.springframework.context": "org.springframework:spring-context",
    "org.springframework.beans": "org.springframework:spring-beans",
    "org.springframework": "org.springframework:spring-core",
    "org.apache.http": "org.apache.httpcomponents:httpclient",
    "okhttp3": "com.squareup.okhttp3:okhttp",
    "retrofit2": "com.squareup.retrofit2:retrofit",
    "io.netty": "io.netty:netty-all",
    "javax.persistence": "javax.persistence:javax.persistence-api",
    "org.hibernate": "org.hibernate:hibernate-core",
    "javax.servlet": "javax.servlet:javax.servlet-api",
    "org.json": "org.json:json",
    "org.projectlombok": "org.projectlombok:lombok",
    "lombok": "org.projectlombok:lombok",
    "io.reactivex": "io.reactivex.rxjava2:rxjava",
    "org.apache.poi": "org.apache.poi:poi",
    "com.amazonaws": "com.amazonaws:aws-java-sdk-core",
    "software.amazon.awssdk": "software.amazon.awssdk:sdk-core",
    "org.yaml.snakeyaml": "org.yaml:snakeyaml",
    "com.opencsv": "com.opencsv:opencsv",
}


def find_pom_files(repo_path: str) -> List[str]:
    pom_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if "pom.xml" in files:
            pom_files.append(os.path.join(root, "pom.xml"))
    return pom_files


def _local(tag: str) -> str:
    """Strip the Maven POM XML namespace from an ElementTree tag name."""
    return tag.rsplit("}", 1)[-1]


def _parse_single_pom(pom_path: str) -> List[Dict[str, str]]:
    """
    Parse one pom.xml's direct <dependencies> (not dependencyManagement,
    and not resolved transitive dependencies — this is a static, offline
    read of what the project itself declares).
    """
    deps: List[Dict[str, str]] = []
    try:
        tree = ET.parse(pom_path)
    except ET.ParseError:
        return deps

    root = tree.getroot()

    properties: Dict[str, str] = {}
    for props_el in root:
        if _local(props_el.tag) == "properties":
            for prop in props_el:
                properties[_local(prop.tag)] = (prop.text or "").strip()

    def resolve_placeholder(value: Optional[str]) -> str:
        if not value:
            return UNKNOWN
        value = value.strip()
        m = re.fullmatch(r"\$\{([^}]+)\}", value)
        if m:
            return properties.get(m.group(1), UNKNOWN)
        return value

    for deps_el in root:
        if _local(deps_el.tag) != "dependencies":
            continue
        for dep_el in deps_el:
            if _local(dep_el.tag) != "dependency":
                continue
            group_id = artifact_id = version = None
            for child in dep_el:
                name = _local(child.tag)
                if name == "groupId":
                    group_id = (child.text or "").strip()
                elif name == "artifactId":
                    artifact_id = (child.text or "").strip()
                elif name == "version":
                    version = resolve_placeholder(child.text)
            if group_id and artifact_id:
                deps.append({
                    "groupId": group_id,
                    "artifactId": artifact_id,
                    "version": version or UNKNOWN,
                })

    return deps


def load_declared_dependencies(repo_path: str) -> List[Dict[str, str]]:
    """
    Parse every pom.xml found under repo_path and return a de-duplicated
    list of {groupId, artifactId, version} the project declares. This is a
    static read of the POM(s) only — it does not run Maven or resolve
    transitive dependencies.
    """
    seen = {}
    for pom_path in find_pom_files(repo_path):
        for dep in _parse_single_pom(pom_path):
            key = (dep["groupId"], dep["artifactId"])
            seen.setdefault(key, dep)
    return list(seen.values())


def extract_package(calls_str: str) -> str:
    """
    Best-effort package extraction from a dotted call string such as
    "org.apache.commons.lang3.StringUtils.trim", using the Java convention
    that package segments are lowercase and the class segment starts the
    first capitalized token: -> "org.apache.commons.lang3".

    Returns "" when no capitalized segment is found (e.g. an unresolved
    "bar.doWork" call where we never learned a real type), since there's
    nothing package-like to resolve in that case.
    """
    segments = calls_str.split(".")
    for i, seg in enumerate(segments):
        if seg[:1].isupper():
            return ".".join(segments[:i])
    return ""


def _longest_prefix_match(package: str, table_keys) -> Optional[str]:
    parts = package.split(".")
    for length in range(len(parts), 0, -1):
        candidate = ".".join(parts[:length])
        if candidate in table_keys:
            return candidate
    return None


def resolve_dependency(calls_str: str, declared_dependencies: List[Dict[str, str]]) -> Optional[str]:
    """
    Resolve an external call string to a dependency identifier.

    Returns:
      - JDK ("jdk") if the package is java.* or javax.*
      - "groupId:artifactId" if resolved via the known-library table or a
        declared pom.xml groupId prefix match
      - "unknown" if the package can't be determined or doesn't match
        anything we know about
    """
    package = extract_package(calls_str)
    if not package:
        return UNKNOWN

    # Known-library table first: some javax.* packages (e.g. javax.persistence,
    # javax.servlet) are real Maven dependencies, not part of the JDK, so a
    # more specific known match must win before the blanket JDK fallback.
    known_match = _longest_prefix_match(package, KNOWN_PACKAGE_PREFIXES)
    if known_match:
        return KNOWN_PACKAGE_PREFIXES[known_match]

    if package == "java" or package.startswith("java.") or package == "javax" or package.startswith("javax."):
        return JDK

    best_group_id = None
    for dep in declared_dependencies:
        group_id = dep["groupId"]
        if package == group_id or package.startswith(group_id + "."):
            if best_group_id is None or len(group_id) > len(best_group_id):
                best_group_id = group_id
                best_dep = dep

    if best_group_id is not None:
        return f"{best_dep['groupId']}:{best_dep['artifactId']}"

    return UNKNOWN
