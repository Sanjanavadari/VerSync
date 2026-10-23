# Person 1: dataset, API diffs, changelogs, failure ground truth

Scripts and outputs for the five Person 1 tasks. Paths are relative to `person1/`.
Scripts that read BUMP expect a checkout at `../dataset/bump`
(`git clone https://github.com/chains-project/bump dataset/bump`). Scripts that call GitHub
need a token: `GITHUB_TOKEN=$(gh auth token)`. Everything uses free sources only
(GitHub API, Maven Central, the Jenkins repository, local `javap`).

## Datasets

| File | Rows | Labels from |
|---|---|---|
| `final_verified_dataset.csv` | 59 (51 BREAKING, 8 SAFE) | BUMP docker reproduction (pre PASS, post FAIL/PASS) |
| `safe_ci_samples.csv` | 45 SAFE | merged bot PR with green CI (Task 1) |
| `combined_dataset.csv` | 104 (51 BREAKING, 53 SAFE) | both; see the `label_source` column |

`final_verified_dataset.csv` is unchanged. Use `combined_dataset.csv` to get more SAFE samples,
and keep `label_source` around so results can be split by label source.

## Task 1: more SAFE samples (`mine_safe_ci_samples.py`)

The 8 original SAFE samples come from projects with no BREAKING sample. That lets a model tell
the classes apart by repository. So the new SAFE samples are mined from the **same
repositories as the BREAKING samples**: 47 reachable repos (docker-adapter was deleted and
google-cloud-java was skipped as too large).

A merged Dependabot/Renovate PR is accepted only if:
- the title names one Maven coordinate with old and new versions (grouped updates are skipped);
- every changed file is a `pom.xml`;
- CI on the PR head is green, with at least one real build check (not just CLA, codecov or
  labelers).

The pre-update commit is the PR's merge base. Its CI is recorded: 41/45 GREEN, 0 red.
`safe_ci_rejections.csv` lists every PR that was looked at and why it was rejected.

**Result:** 45 SAFE samples from 24 repos (at most 2 per repo): 33 Dependabot, 12 Renovate.

Caveats:
- The labels come from CI, not from a docker rebuild.
- The new SAFE updates are mostly minor/patch bumps merged in 2025–26. The BUMP BREAKING
  samples are older and more often major bumps, so SemVer type and date still differ between
  the classes.
- DIGs for the new samples still need to be built at `pre_commit_sha` (Person 2).

## Task 2: API diffs (`api_diff_extractor.py`)

`python api_diff_extractor.py --csv combined_dataset.csv` writes one JSON per update to
`api_diffs/` and one row per sample to `api_diff_summary.csv`.

Changes from the first version:
- Downloads from Maven Central, repo.jenkins-ci.org and the Groovy repository, with timeouts
  and retries.
- Packaging is read from the POM. Before, any artifactId containing "plugin" was skipped,
  which silently dropped `maven-plugin-api`.
  - `jar` and friends: the methods are diffed.
  - `hpi`/`jpi` Jenkins plugins: the plugin is diffed together with its bundled `WEB-INF/lib`
    jars.
  - `pom` (BOMs): there is no API, so managed dependency versions are diffed instead
    (`NOT_APPLICABLE_POM`).
- Every diff also compares the direct dependencies of the two POMs, and lists removed and added
  classes.
- Summary columns: `status`, `classification`, relative `diff_json` paths.

Coverage of the 59 verified samples: **59/59** (it was 35, plus 2 that were empty). This
includes all 8 SAFE samples, so having a diff no longer gives away the label. For the 104
combined samples, see the status counts printed by the script.

## Task 3: changelogs (`mine_sample_changelogs.py`)

Writes one JSON per sample to `changelogs/samples/` and an index to
`changelogs/sample_changelog_index.csv`.

Sources, all collected:
- the bot PR body (its release notes, changelog and commits sections);
- GitHub releases whose tag version falls in (old, new];
- the CHANGELOG section between the two versions.

The library's repository comes from the POM `<scm>`. BUMP's `githubRepoSlug` is only a
fallback, because it is sometimes wrong (mina-core pointed to `apache/mina-sshd`).

| | TEXT_EXTRACTED | COMMITS_ONLY | LINK_ONLY | MISSING |
|---|---|---|---|---|
| BREAKING (51) | 37 | 10 | 4 | 0 |
| SAFE (53) | 46 | 4 | 0 | 3 |

Before this, 4 of the 59 verified samples had extracted text and none of the SAFE samples had
an entry. The older `mine_changelogs.py` and `changelogs/changelog_index.csv` are kept for
reference.

## Task 4: failure ground truth (`extract_failure_ground_truth.py`)

Parses all 571 BUMP reproduction logs into `failure_ground_truth.csv/.json`.

New in this version:
- More test-report formats: JUnit 5 parameterized, JUnit 4 `method(Class)`, Surefire 3
  `Class#method`, and carrotsearch randomizedtesting.
- `-Werror` files.
- A `failed_dependencies` column for build-configuration failures that have no source
  location: upper-bound and convergence errors, banned dependencies, dependency-lock
  mismatches, and Jenkins "requires Jenkins X".
- `log_path` is now relative.

For the 51 BREAKING samples in the dataset:
- 40 have files, functions or tests (was 33). All 6 TEST_FAILURE samples and the `-Werror`
  sample are now localized.
- The other 11 are enforcer or dependency-lock failures. All 11 now have
  `failed_dependencies`; for the 7 Jenkins plugins this is the dependency that "requires
  Jenkins X".
- No dataset sample is left without any localization.

Across all 571 logs, 469 have a code location, 138 have failed dependencies and 29 have
nothing.

## Task 5: Python generalization set (`verify_python_samples.py`)

`mine_python_dataset.py` collected PRs with green CI ("candidates") and one hand-seeded breaking
example. `verify_python_samples.py` labels every row using a **CI differential** between the
pre-update commit (PR merge base) and the post-update commit (PR head). The PR must change only
dependency manifests.

- **BREAKING**: a build/test check that passed before the update fails after it.
- **SAFE**: everything is green both before and after.
- **UNVERIFIED**: anything else.

It also mines new BREAKING samples (Dependabot/Renovate PRs in Python repos with failing CI,
at most 2 per repo). `failing_checks` lists exactly which checks flipped, so flaky ones can be
reviewed. Like Task 1, this is CI evidence, not a local rebuild.

**Result (51 rows):**
- **25 BREAKING** from 21 repos. 20 have a test check that flipped; 5 have only a build check
  that flipped.
- **15 SAFE**: these are 15 of the earlier green candidates, now also confirmed green before
  the update.
- **11 UNVERIFIED**:
  - 9 PRs that change more than dependency manifests (mostly GitHub Actions or tool bumps the
    first miner picked up);
  - 1 where CI was already red before the update;
  - the hand-seeded `requests`/urllib3 row (its PR is not a single dependency bump).

  Leave UNVERIFIED rows out of evaluation.

Failures in security scans, audits, Docker pushes and deployments are not counted as
breaking.

## Known gaps

- New SAFE samples have no DIG yet. Their `pre_commit_sha` is the commit to build.
- CI-derived labels (Task 1, Task 5) are weaker than docker reproduction.
- 11 BREAKING samples (enforcer/lock) are localized only to a dependency, not to code, by nature.
