# graph-model

Method-level call-graph extractor for Java repositories, built for VerSync's
dependency-update risk prediction pipeline. Given a cloned Java repo, it
produces an edge list of `(file, function, called_api_or_function, external)`
tuples, distinguishing calls to code inside the project from calls into
imported third-party/JDK libraries.

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

## Setup

```bash
cd graph-model
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `javalang==0.13.0`.

## Usage

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
    {"file": "src/main/java/org/apache/commons/lang3/StringUtils.java", "function": "abbreviate", "calls": "StringUtils.isEmpty", "external": false},
    {"file": "src/main/java/org/apache/commons/lang3/StringUtils.java", "function": "abbreviate", "calls": "java.lang.String.length", "external": true}
  ]
}
```

The script prints a summary (files found/parsed/failed, edge counts) to
stdout, and lists any files it could not parse to stderr.

## Parsing approach

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

## Limitations

- **Static imports**: handled for the common cases — `import static X.Y;`
  correctly attributes a bare `Y()` call to `X.Y` (verified: `assertEquals()`
  resolves to `org.junit.Assert.assertEquals`). `import static X.*;` can't
  know which method names actually come from `X` without full symbol
  resolution, so an unresolved bare call falls back to guessing the first
  such wildcard class — this can be wrong if a file has more than one static
  wildcard import.
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
- **Constructor calls (`new Foo()`)** are not currently emitted as edges —
  only `MethodInvocation` nodes are tracked, not `ClassCreator` nodes.
- **Modern Java syntax**: javalang predates `record`, sealed classes, pattern
  matching for `switch`, and text blocks in places; files using such syntax
  either fail to parse (skipped, logged to stderr) or parse with gaps. On a
  626-file real-world run (`apache/commons-lang`, depth-1 clone), 4 files
  failed to parse this way — javalang's `JavaSyntaxError` also often carries
  an empty message, so failures are reported by file path only, not by
  reason.

## Verified test run

```bash
git clone --depth 1 https://github.com/apache/commons-lang.git /tmp/commons-lang
python extract_call_graph.py /tmp/commons-lang commons-lang-graph.json
```

Result: 626 `.java` files found, 622 parsed successfully (4 failed on
modern/edge-case syntax javalang doesn't support), producing 79,116 edges
(33,578 internal / 45,538 external).

A trimmed 20-edge sample of the real output (from `StringUtils.java`) is
committed as [`sample_output.json`](sample_output.json) for reference — the
full 79k-edge output was ~16 MB and isn't checked in.
