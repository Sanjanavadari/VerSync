# graph-model

Static analysis for Java repositories, built for VerSync's dependency-update
risk prediction pipeline. Given a cloned Java repo, this folder can:

1. **`extract_call_graph.py`** — build a method-level call-graph edge list,
   distinguishing internal calls (within the project) from external calls,
   and, for external calls, resolve *which Maven dependency* provides the
   called API where reasonably possible.
2. **`build_dig.py`** — turn that edge list into an explicit
   **Dependency-Impact Graph (DIG)**: `FILE → FUNCTION → API → DEPENDENCY`
   nodes and edges, so downstream consumers (the eval/CLI pipeline, and
   eventually a risk model) can walk from a file to the dependencies it
   actually touches, or from a dependency back to every file that could be
   affected if it changes.

## Why javalang instead of JavaParser

Two viable options for parsing Java from this pipeline:

- **JavaParser** (Java library): more accurate, actively maintained, has a
  real symbol solver that can resolve types precisely (including across
  modules with a configured classpath). The cost is setup: it needs a JDK,
  Maven/Gradle, and either a small Java driver program invoked as a
  subprocess from Python, or a JVM bridge (e.g. `jpype`/`py4j`). More correct
  long-term, slower to stand up.
- **javalang** (pure Python): `pip install javalang` and you're parsing in
  minutes, no JVM/build tooling involved, and the AST is a normal Python
  object tree that's trivial to walk from this script. The cost is accuracy:
  it's a hand-written, community-maintained parser last updated for roughly
  Java 8 syntax, it has no real type solver, and it can fail outright on some
  modern syntax (see Limitations).

For a first working prototype, **javalang wins on speed-to-value** — this
whole extractor, including tests against a real repo, was built without
touching a JDK. If accuracy on modern Java syntax (records, pattern matching,
sealed classes) or precise generic-type resolution becomes a blocker, the
natural next step is porting this same logic to a small JavaParser-based Java
CLI that dumps its own JSON, called from Python via `subprocess`.

The same reasoning applies to **dependency resolution** below: we don't run
Maven or read jars from a local `~/.m2` repository (no JDK/Maven/network
dependency), we read `pom.xml` as plain XML and cross-reference it with a
curated table of well-known libraries. Less precise than real classpath
resolution, but deterministic, explainable, and works offline.

## Setup

```bash
cd graph-model
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `javalang==0.13.0`. `dependency_resolver.py` and
`build_dig.py` use only the Python standard library — no new dependencies.

## Usage

### 1. Call graph with dependency resolution

```bash
python extract_call_graph.py <path-to-cloned-repo> <output.json>
```

Example:

```bash
git clone --depth 1 https://github.com/apache/commons-lang.git /tmp/commons-lang
python extract_call_graph.py /tmp/commons-lang commons-lang-graph.json
```

Output shape:

```json
{
  "repo": "commons-lang",
  "edges": [
    {"file": "src/main/java/org/apache/commons/lang3/StringUtils.java", "function": "abbreviate", "calls": "StringUtils.isEmpty", "external": false, "dependency": null},
    {"file": "src/main/java/org/apache/commons/lang3/StringUtils.java", "function": "abbreviate", "calls": "java.lang.String.length", "external": true, "dependency": "jdk"}
  ]
}
```

`dependency` is the new field (Task 1). It is:

- `null` when `external` is `false` (not applicable — the call is internal),
- `"jdk"` for `java.*`/`javax.*` calls,
- `"<groupId>:<artifactId>"` (e.g. `"com.fasterxml.jackson.core:jackson-databind"`)
  when resolved via the known-library table or the project's own `pom.xml`,
  or
- `"unknown"` when the call is external but we can't determine a dependency
  — we never invent one.

We kept the existing field names (`file`, `function`, `calls`, `external`)
exactly as they were rather than renaming `calls` → `api` (as a strawman
schema in the task brief suggested), since nothing about the existing,
already-tested extractor needed to change — `dependency` is purely additive.
See [Dependency-Impact Graph](#dependency-impact-graph-dig--build_digpy-task-2)
below for the `api`-labeled node the same information ends up as, once it's
reshaped into a graph.

The script prints a summary (files found/parsed/failed, edge counts, and
now dependency-resolution counts) to stdout, and lists any files it could
not parse to stderr.

### 2. Dependency-Impact Graph (DIG)

```bash
python build_dig.py <path-to-cloned-repo> <output.json>
```

This reuses `extract_call_graph.py`'s parsing pipeline directly (same
javalang parsing, symbol table, and dependency resolution — nothing is
re-implemented) and reshapes the resulting edges into an explicit graph.

Output shape:

```json
{
  "repo": "commons-lang",
  "nodes": [
    {"id": "file:src/main/java/org/apache/commons/lang3/StringUtils.java", "type": "FILE", "name": "src/main/java/org/apache/commons/lang3/StringUtils.java"},
    {"id": "function:src/main/java/org/apache/commons/lang3/StringUtils.java::abbreviate", "type": "FUNCTION", "name": "abbreviate"},
    {"id": "api:java.lang.String.length", "type": "API", "name": "java.lang.String.length"},
    {"id": "dependency:jdk", "type": "DEPENDENCY", "name": "jdk"}
  ],
  "edges": [
    {"source": "file:...StringUtils.java", "target": "function:...StringUtils.java::abbreviate", "type": "CONTAINS"},
    {"source": "function:...StringUtils.java::abbreviate", "target": "api:java.lang.String.length", "type": "CALLS"},
    {"source": "api:java.lang.String.length", "target": "dependency:jdk", "type": "PROVIDED_BY"}
  ]
}
```

Two deliberate deviations from the schema sketched in the task brief, both
to keep the graph correct and inspectable:

- **FUNCTION node ids are scoped by file** (`function:<file>::<name>`), not
  just by name — plain `function:process()` ids would collide across every
  file that happens to declare a same-named method.
- **Internal calls are included too**, as `FUNCTION --CALLS--> FUNCTION`
  edges (not just the external `FUNCTION → API → DEPENDENCY` chain), and —
  importantly — an internal call resolves to the *same* function node that
  function's own `CONTAINS`/`CALLS` edges use (via a class→declaring-file
  map built once per run), not a disconnected duplicate. This is what makes
  multi-hop chains like `testX() → helperMethod() → ObjectMapper.readValue()
  → jackson-databind` walkable in one traversal instead of dead-ending at an
  orphan node. This was caught and fixed during testing — see
  [`test_extractor.py`](test_extractor.py)'s `test_build_dig_chain_connectivity`.

## Dependency resolution approach

`dependency_resolver.py` maps an external call's fully-qualified-ish string
(e.g. `com.fasterxml.jackson.databind.ObjectMapper.readValue`) to a Maven
dependency using two signals, checked in order:

1. **A curated table** (`KNOWN_PACKAGE_PREFIXES`, ~40 entries) of
   package-prefix → `groupId:artifactId` for common libraries whose package
   names don't reliably derive from their Maven coordinates (e.g. Guava's
   groupId is `com.google.guava` but its packages are `com.google.common.*`;
   Jackson's three core artifacts share the `com.fasterxml.jackson.core`
   groupId but live in three different packages). Longest matching prefix
   wins. This table is small and hand-maintained by design — extend it as
   new libraries come up.
2. **The project's own declared `pom.xml` dependencies**, as a fallback: if
   a call's package starts with a declared dependency's `groupId` (a
   convention many libraries *do* follow, e.g.
   `org.apache.commons:commons-lang3` → packages under
   `org.apache.commons.lang3.*`), that dependency wins. Every `pom.xml`
   found under the repo is parsed (plain XML, namespace-stripped, with
   `${property}` placeholders resolved from that same POM's
   `<properties>`) — this is a static read only; we do not invoke Maven or
   resolve transitive dependencies.

`java.*`/`javax.*` calls are labeled `"jdk"` — but only *after* the known
table is checked, since some `javax.*` packages (`javax.persistence`,
`javax.servlet`) are real Maven dependencies, not part of the JDK. (An
earlier version of this checked JDK first; `test_extractor.py` caught the
ordering bug before it shipped.)

If neither signal matches, the result is `"unknown"` — per the task's
explicit instruction, we do not guess.

## Testing

```bash
python test_extractor.py
```

Dependency-free (uses only `unittest`-style plain assertions, matching
`eval-system/evaluator.py`'s no-third-party-deps style), covering:

- `dependency_resolver.py`: known-table matches, JDK detection (including
  the `javax.persistence` non-JDK case), `pom.xml` groupId fallback,
  `pom.xml` property-placeholder resolution, and honest `"unknown"` when
  there's no signal.
- `extract_call_graph.py`: an end-to-end run (via subprocess, against a
  synthetic 1-file/1-pom sample repo) asserting the exact `dependency`
  value for a known-table match, a JDK call, and an internal call (`null`).
- `build_dig.py`: an end-to-end run asserting node/edge counts, that the
  `FILE → FUNCTION → API → DEPENDENCY` chain is actually present, and the
  internal-call connectivity fix described above.

All 8 tests pass as of this writing.

## Parsing approach (call graph)

1. **Pass 1 — project symbol table.** Every `.java` file is parsed once with
   javalang. From the successfully-parsed files, we collect every declared
   class/interface/enum simple name (`internal_classes`) and, per class, the
   set of method names it declares (`internal_methods_by_class`). This is
   the project's own vocabulary — anything not in it is assumed external.
2. **Pass 2 — per-method call extraction.** For each method/constructor
   declaration, we:
   - Build a **variable-type map** combining the enclosing class's fields,
     the method's parameters, and its local variable declarations (name →
     declared type simple name).
   - Build an **import map** from the file's `import` statements: concrete
     imports (`import org.apache.commons.lang3.StringUtils;`) map simple
     name → fully-qualified name; `import static X.Y;` and
     `import static X.*;` are tracked separately so bare calls can be
     attributed to the right external class.
   - Walk every `MethodInvocation` inside the method body (this includes
     invocations nested in lambdas, `if`/`for`/`try` blocks, etc., but
     explicitly **excludes** invocations that belong to a nested/local/
     anonymous class's own method — those are attributed to that inner
     method instead, so nothing is double-counted).
   - For each call, resolve the qualifier (the part before the last `.`) in
     this order: no qualifier / `this` → look at static imports first, then
     internal methods on the enclosing class → qualifier is a known
     variable/field with a tracked type → qualifier matches an internal
     class name (static call) → qualifier matches an imported class →
     qualifier is a common `java.lang` type → otherwise, default to
     "external" (in practice, most calls we can't resolve turn out to be on
     JDK/library types like `List`/`Map`/`StringBuilder`).
   - **(New)** If the call was classified external, resolve its dependency
     via `dependency_resolver.py` (memoized per unique call string, so a
     79k-edge repo only resolves each distinct API once).

## Limitations

- **Static imports**: handled for the common cases — `import static X.Y;`
  correctly attributes a bare `Y()` call to `X.Y` (verified: `assertEquals()`
  resolves to `org.junit.Assert.assertEquals`). `import static X.*;` can't
  know which method names actually come from `X` without full symbol
  resolution, so an unresolved bare call falls back to guessing the first
  such wildcard class — this can be wrong if a file has more than one static
  wildcard import.
- **Static imports of a project's own helper methods get misclassified as
  external** — a real, observed case: `resolve_call`'s bare-call handling
  checks a file's static imports *before* checking whether the imported
  class is actually one of the project's own (`internal_classes`), so
  `import static org.apache.commons.lang3.LangAssertions.assertX;` from a
  commons-lang test helper class gets marked `external: true`. Dependency
  resolution then (correctly, given that misclassification) maps its
  package back to the project's own `pom.xml` groupId, producing a
  "dependency on itself" — 1,552 of the 45,955 external edges (~3.4%) in
  the commons-lang run below are exactly this pattern. This is a
  pre-existing internal/external classification limitation (not something
  Task 1 changed), surfaced more visibly now that dependency resolution
  exists. Fixing it would mean checking `internal_classes` before
  `static_method_map` in `resolve_call` — left as a follow-up rather than
  changed here, to avoid an unscoped rewrite of the already-tested
  classifier.
- **Method overloading**: not disambiguated. `calls`/`function` are simple
  method names with no parameter-type signature, so `foo(int)` and
  `foo(String)` collapse into the same edge name.
- **Lambda expressions**: calls inside a lambda body are attributed to the
  lambda's *enclosing* method (there's no separate node for "the lambda
  itself" in this graph) — verified against a synthetic test case.
- **Generics / type inference**: variable types are taken from their
  declared type name only (e.g. `List<String> items` → `List`); we do not
  resolve generic type parameters, wildcard-imported types, or fields
  inherited from an external superclass.
- **`super.method()` calls**: recorded but not resolved to the actual
  superclass (we don't do full class-hierarchy resolution), so they're
  labeled `EnclosingClass(super).method` rather than the true parent type.
- **Class name collisions**: internal-class detection is by simple name, so
  two unrelated classes named e.g. `Builder` in different packages are
  indistinguishable — a false "internal" match is possible in large repos.
  The DIG's cross-file function-node resolution (`class_to_file` map) has
  the same limitation: it keeps whichever file it saw the class name in
  first.
- **Constructor calls (`new Foo()`)** are not currently emitted as edges —
  only `MethodInvocation` nodes are tracked, not `ClassCreator` nodes.
- **Modern Java syntax**: javalang predates `record`, sealed classes, pattern
  matching for `switch`, and text blocks in places; files using such syntax
  either fail to parse (skipped, logged to stderr) or parse with gaps.
  javalang's `JavaSyntaxError` also often carries an empty message, so
  failures are reported by file path only, not by reason.
- **Dependency resolution is heuristic, not classpath-accurate**: no jar is
  ever inspected. The known-library table is hand-maintained and covers
  common cases (~40 libraries), not every library in the Maven ecosystem.
  The `pom.xml` groupId-prefix fallback assumes a naming convention that
  not all libraries follow. Multi-module repos are handled by parsing every
  `pom.xml` found under the repo root and merging their declared
  dependencies, without modeling which module actually owns which source
  file. Dependency *versions* are read from `pom.xml` but not currently
  used in the resolved `dependency` string (only `groupId:artifactId`) —
  version-aware resolution (useful for "is this the specific dependency
  version being updated?") is a natural next step.

## Verified test runs

Run against two real, unmodified Java/Maven repositories:

**`apache/commons-lang`** (depth-1 clone, 629 `.java` files):

```bash
git clone --depth 1 https://github.com/apache/commons-lang.git /tmp/commons-lang
python extract_call_graph.py /tmp/commons-lang commons-lang-graph.json
python build_dig.py /tmp/commons-lang commons-lang-dig.json
```

- `extract_call_graph.py`: 624/629 files parsed, 79,995 edges (34,040
  internal / 45,955 external). **38,159/45,955 (83%) external edges
  resolved** to a known dependency or `"jdk"`; 7,796 left `"unknown"`
  (mostly unresolved-variable-type calls where we never learn a real type,
  e.g. `oos.writeObject(...)`, plus a handful of javalang parsing edge
  cases with primitive-type qualifiers like `boolean.getClass`).
- `build_dig.py`: 11,292 nodes (624 FILE / 9,193 FUNCTION / 1,469 API / 6
  DEPENDENCY) and 35,459 edges (7,313 CONTAINS / 26,677 CALLS / 1,469
  PROVIDED_BY). Dependency nodes found: `jdk`, `unknown`,
  `org.apache.commons:commons-lang3` (see the static-import limitation
  above), `org.junit.jupiter:junit-jupiter-api`, `org.mockito:mockito-core`,
  `org.easymock:easymock`.

**`spring-projects/spring-petclinic`** (depth-1 clone, 50 `.java` files) —
chosen as a second repo specifically for more dependency diversity
(Spring, JUnit 5, Mockito, AssertJ, Testcontainers) than a
single-library-focused repo like commons-lang:

```bash
git clone --depth 1 https://github.com/spring-projects/spring-petclinic.git /tmp/spring-petclinic
python extract_call_graph.py /tmp/spring-petclinic petclinic-graph.json
python build_dig.py /tmp/spring-petclinic petclinic-dig.json
```

- `extract_call_graph.py`: 49/50 files parsed, 1,715 edges (814 internal /
  901 external). **828/901 (92%) external edges resolved.**
- `build_dig.py`: 534 nodes (49 FILE / 306 FUNCTION / 168 API / 11
  DEPENDENCY) and 1,367 edges. Dependency nodes found: `jdk`, `unknown`,
  `org.springframework.boot:spring-boot`, `org.springframework:spring-core`,
  `org.springframework:spring-context`, `org.springframework:spring-web`,
  `org.junit.jupiter:junit-jupiter-api`, `org.mockito:mockito-core`,
  `org.hamcrest:hamcrest`, `org.assertj:assertj-core`,
  `org.testcontainers:testcontainers-junit-jupiter`.

A representative, verified chain from this run:

```
FILE:       src/test/.../PetClinicConcurrencyTests.java
  → FUNCTION: testDuplicatePetNameRaceConditionIsBlocked
    → API:      org.springframework.web.client.RestTemplate.postForEntity
      → DEPENDENCY: org.springframework:spring-web
```

...and a cross-file internal chain confirming the connectivity fix (the
target function node is the *same* node `OwnerRepository.java`'s own
declaration uses, in a different file than the caller):

```
FUNCTION: PetClinicConcurrencyTests.java::testDuplicatePetNameRaceConditionIsBlocked
  --CALLS--> FUNCTION: OwnerRepository.java::findById
```

Trimmed, real (not fabricated) samples are committed for reference — full
outputs weren't checked in due to size (commons-lang's DIG alone is ~10MB):

- [`sample_output.json`](sample_output.json) — 7 representative edges from
  the petclinic run (2 internal, 5 external across 5 different resolved
  dependency values including `unknown`).
- [`dig_sample_output.json`](dig_sample_output.json) — the full local DIG
  subgraph (41 nodes / 53 edges) rooted at `PetClinicConcurrencyTests.java`
  from the same run, including the cross-file chain above.

## What's next

This is Task 1 + Task 2 of the graph-model workstream: static
call/dependency extraction and a first explicit DIG representation. Not
implemented yet (by design — see the task brief): the ML risk model, GNN or
other graph-based propagation, changelog mining, or behavioral analysis.
The natural next integration step is having `eval-system/analyzer.py`
consume a DIG for the repo being analyzed and use its `DEPENDENCY` nodes to
find real impacted files/functions for the dependency being updated,
replacing today's mock `_mock_impacted_files`.
