# DIG build report

## Update 2026-10-26: 102 of 104 combined samples

All DIGs were rebuilt with the fixed extractor/resolver, and DIGs were added for the 45 CI-labelled SAFE
samples in `person1/combined_dataset.csv`:

```
python eval-system/batch_build_digs.py --rebuild --no-head-fallback --only <all projects except google-cloud-java>
```

| | Samples |
|---|---|
| Combined dataset | 104 (51 BREAKING, 53 SAFE) |
| **DIGs** | **102 (49 BREAKING, 53 SAFE)**, all at the intended pre-update commit (0 HEAD fallbacks) |
| DIG files | 97: 5 CI-labelled samples share their pre-update commit with another sample of the same project, so they share its DIG |
| Not built | `docker-adapter` (repository deleted), `google-cloud-java` (skipped: over the build timeout, and its update is a BOM) |

- CI-labelled samples have no BUMP `source_json`, so their DIG is named `<project>_<pre_commit_sha>.json`
  and their report row has `source_json = ci:<pre_commit_sha>`.
- Extractor/resolver changes (see `graph-model/`): constructor calls and annotations on imported library
  types are recorded as API uses; `log4j-api` vs `log4j-core` and `junit-jupiter-params` are mapped
  correctly; among several declared artifacts of one groupId, the one whose name matches the package wins.
- The updated dependency is now a DEPENDENCY node in **31 of the original 57 DIGs** (was 26; none lost),
  43 of all 102. The largest DIG is 41 MB (causeway).

The sections below describe the earlier 57-DIG build.


Per-sample details are in [`dig_build_report.csv`](dig_build_report.csv). Each DIG also has a
`build_metadata` block that records which commit was analyzed.

## Summary

| | Samples |
|---|---|
| In `person1/final_verified_dataset.csv` | 59 (51 BREAKING, 8 SAFE) |
| DIGs before this work | 19 (8 SAFE + the first 11 BREAKING rows) |
| **DIGs now** | **57 (49 BREAKING, 8 SAFE)** |
| Built at the intended pre-update commit | 57 / 57 (8 `parent_commit`, 49 `breaking~1`, **0 HEAD fallbacks**) |
| Not buildable | 2 (both BREAKING) |

## Why only 19 existed: failure causes for the 40 missing samples

| Cause | Count | Fixed? |
|---|---|---|
| **Never attempted**: the original batch run stopped after the first 20 rows in processing order (8 SAFE, then BREAKING in CSV order), most likely from `--limit` or an interruption. Person 3's original script, re-run unchanged, builds them with identical node/edge counts (checked on camunda-platform-7-mockito and lithium). | 36 | Yes, built |
| **Failed once, not reproducible**: publish-over-ssh-plugin (row 20 in processing order) is missing from the original 19, but the unchanged original script builds it fine on macOS (989 nodes / 2,311 edges, same as the new build). The cause was most likely transient (network) or specific to the Windows machine; no log was kept. | 1 | Yes, built |
| **Repository moved**: `sonatype-nexus-community/polyglot-maven` no longer exists; the BUMP breaking commit is in `takari/polyglot-maven`. | 1 | Yes, via `REPO_URL_OVERRIDES` in `batch_build_digs.py` |
| **Repository deleted**: `artipie/docker-adapter` returns 404; commit `13fd75e2` is not in `artipie/artipie` either. | 1 | **No** |
| **Too large / build timeout**: `google-cloud-java` is a 57,430-file monorepo. It hit the 900 s build timeout, and a 3,600 s retry was stopped. The DIG would also go over GitHub's 100 MB file limit, and the updated artifact (`google-cloud-pubsub-bom`) is a BOM that code never calls, so it would add no graph signal. | 1 | **No** |

Causes that did **not** occur:
- **Clone failures** other than the two above: none.
- **Missing pom.xml / multi-module layouts**: none. `find_pom_files` already walks every module's `pom.xml`.
- **Repos failing on javalang**: none. Parse errors are already skipped per file.

Across the 57 built DIGs:
- 100 of 15,969 `.java` files (0.6%) could not be parsed and were skipped.
- 16 repos had at least one skipped file. The worst were avans (31/129), junit-quickcheck (22/507) and causeway (12/5,544).
- Recorded causes: javalang 0.13 does not support Java 9+ syntax (records, switch expressions, text blocks), and Maven archetype templates (`#set(...)`) are not Java.

## Fixes made

1. `graph-model/build_dig.py`: if edge extraction fails on one file, that file is skipped instead of failing the whole repo.
2. `eval-system/batch_build_digs.py`:
   - fetches `pull/<n>/head`, so BUMP breaking commits are reachable even after the bot branch is deleted;
   - records `checkout_mode` (it used to fall back to HEAD silently, which would have analyzed today's code instead of the pre-update state);
   - adds a per-sample timeout and the CSV report;
   - adds `--rebuild`, `--only`, `--no-head-fallback`, `--git-timeout`, `--build-timeout`;
   - adds `REPO_URL_OVERRIDES`.

## Consistency with the original 19 DIGs

All 19 were rebuilt:
- 10 match exactly on node and edge counts by type.
- 8 differ by under 1% in FUNCTION nodes. The likely cause is Windows vs macOS directory walk order: internal-call resolution keeps the first class seen with a given simple name.
- 1 (micro-migration `00b60810`) has 95 files instead of 90 at the same parent commit, most likely a Windows checkout difference.

The original DIGs stored Windows (`\`) paths; the rebuilt ones use `/`.

## Known limitation for features (not a build failure)

The updated dependency is found as a DEPENDENCY node in only **26 of 57** DIGs. The resolver maps a call to a dependency only when its package matches a `groupId` declared in `<dependencies>`. So these are invisible to the graph:
- BOMs and `<dependencyManagement>` entries;
- Maven/build plugins;
- runtime-only artifacts;
- dependencies whose package differs from their `groupId`.

Also, a median of about 30% of API calls resolve to `dependency:unknown`.
