import json
import csv
import os
import subprocess
import shutil
import sys
import stat
import time
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BUMP_DIR = Path("dataset/bump")

SOURCE_DIRS = {
    "benchmark": BUMP_DIR / "data" / "benchmark",
    "unsuccessful": BUMP_DIR / "data" / "unsuccessful-reproductions",
}

WORK_DIR = Path("person1") / "task2_repos"
LOG_DIR = Path("person1") / "task2_logs"

OUTPUT_CSV = Path("person1") / "task2_verification_results.csv"
OUTPUT_JSON = Path("person1") / "task2_verification_results.json"

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.32.101-hotspot"
MAVEN = r"C:\Users\siva_\Desktop\apache-maven-3.9.16\bin\mvn.cmd"

# 25 minutes per Maven build
MAVEN_TIMEOUT_SECONDS = 1500

# Git operation limits
GIT_CLONE_TIMEOUT_SECONDS = 600
GIT_OP_TIMEOUT_SECONDS = 180

RESULT_FIELDS = [
    "project",
    "repo_url",
    "dependency_group_id",
    "dependency_artifact_id",
    "old_version",
    "new_version",
    "java_version",
    "failure_category",
    "pre_update_status",
    "post_update_status",
    "classification",
    "pre_exit_code",
    "post_exit_code",
    "breaking_commit",
    "parent_commit",
    "pre_log",
    "post_log",
    "source_json",
]

IS_WINDOWS = os.name == "nt"


# ============================================================
# PROCESS HELPERS
# ============================================================

def kill_process_tree(pid):
    """
    Kill the complete process tree.

    On Windows, Maven normally launches Java through cmd.exe.
    Killing only the wrapper can leave java.exe running.
    taskkill /T kills the entire tree.
    """

    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except Exception:
            pass

    else:
        try:
            subprocess.run(
                ["pkill", "-TERM", "-P", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except Exception:
            pass


# ============================================================
# MAVEN BUILD
# ============================================================

def run_build(command, cwd, timeout, log_path):
    """
    Run Maven directly without shell=True.

    On Windows this allows Python to control the actual mvn.cmd
    process instead of launching it through an extra cmd.exe wrapper.
    """

    env = os.environ.copy()

    env["JAVA_HOME"] = JAVA_HOME
    env["PATH"] = JAVA_HOME + r"\bin;" + env.get("PATH", "")

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        if IS_WINDOWS
        else 0
    )

    print(
        "  running:",
        " ".join(f'"{x}"' if " " in x else x for x in command),
        flush=True
    )

    start_time = time.time()

    try:

        with open(
            log_path,
            "w",
            encoding="utf-8",
            errors="replace"
        ) as log_file:

            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
                shell=False,
            )

            try:

                exit_code = process.wait(
                    timeout=timeout
                )

            except subprocess.TimeoutExpired:

                print(
                    f"  TIMEOUT after {timeout}s "
                    f"- killing process tree "
                    f"(pid {process.pid})",
                    flush=True
                )

                kill_process_tree(
                    process.pid
                )

                try:
                    process.wait(
                        timeout=15
                    )
                except subprocess.TimeoutExpired:
                    pass

                duration = (
                    time.time()
                    - start_time
                )

                print(
                    f"  killed after "
                    f"{duration:.0f}s total",
                    flush=True
                )

                return "TIMEOUT", None

    except Exception as exc:

        print(
            f"  Maven execution error: {exc}",
            flush=True
        )

        return "ERROR", None

    duration = (
        time.time()
        - start_time
    )

    status = detect_build_status(
        log_path
    )

    print(
        f"  finished in {duration:.0f}s, "
        f"exit={exit_code}, "
        f"status={status}",
        flush=True
    )

    return status, exit_code


# ============================================================
# MAVEN RESULT DETECTION
# ============================================================

def detect_build_status(log_path):
    """
    Read the Maven log and use the LAST exact Maven reactor
    BUILD SUCCESS / BUILD FAILURE line.

    This avoids trusting the Docker/Maven process exit code alone.
    """

    last_status = None

    try:
        with open(
            log_path,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as log_file:

            for line in log_file:

                line = line.strip()

                if line == "[INFO] BUILD SUCCESS":
                    last_status = "PASS"

                elif line == "[INFO] BUILD FAILURE":
                    last_status = "FAIL"

    except FileNotFoundError:
        return "ERROR"

    return last_status if last_status else "ERROR"


# ============================================================
# GIT HELPERS
# ============================================================

def run_git(args, cwd=None, timeout=GIT_OP_TIMEOUT_SECONDS):
    """
    Execute Git with:

    - credential prompts disabled
    - fixed timeout
    - captured output
    """

    env = os.environ.copy()

    # Never allow Git to wait for credentials interactively.
    env["GIT_TERMINAL_PROMPT"] = "0"

    command = ["git"]

    if cwd is not None:
        command.extend(["-C", str(cwd)])

    command.extend(args)

    try:

        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired:

        print(
            f"  Git command timed out after "
            f"{timeout}s: {' '.join(command)}",
            flush=True
        )

        return None


# ============================================================
# REPOSITORY CLEANUP
# ============================================================

def remove_repository(repo_dir, retries=2):
    """
    Remove an existing working repository.

    Never terminates the whole batch because of a locked file.
    """

    if not repo_dir.exists():
        return True

    def remove_readonly(func, path, exc_info):

        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)

        except Exception:
            pass

    for attempt in range(retries + 1):

        try:

            try:
                shutil.rmtree(
                    repo_dir,
                    onexc=remove_readonly
                )

            except TypeError:
                # Compatibility fallback
                shutil.rmtree(
                    repo_dir,
                    onerror=remove_readonly
                )

            return True

        except PermissionError:

            if attempt < retries:

                print(
                    "  Repository is locked; "
                    "retrying in 5 seconds...",
                    flush=True
                )

                time.sleep(5)

            else:

                print(
                    "  Could not remove repository "
                    "(still locked) - INVALID case.",
                    flush=True
                )

                return False

        except Exception as exc:

            print(
                f"  Repository removal failed: {exc}",
                flush=True
            )

            return False

    return False


# ============================================================
# METADATA HELPERS
# ============================================================

def get_repo_url(case):
    """
    Determine the project repository.

    Preferred:
        BUMP PR URL

    Fallback:
        updatedDependency.githubRepoSlug
    """

    url = case.get("url", "")

    if "/pull/" in url:

        return (
            url.split("/pull/")[0]
            + ".git"
        )

    slug = (
        case
        .get("updatedDependency", {})
        .get("githubRepoSlug")
    )

    if slug and slug != "Repository not found":

        return f"https://github.com/{slug}.git"

    return None


def get_pull_number(case):

    url = case.get("url", "")

    if "/pull/" not in url:
        return None

    try:

        return int(
            url.split(
                "/pull/",
                1
            )[1].split(
                "/",
                1
            )[0]
        )

    except (ValueError, IndexError):

        return None


# ============================================================
# BREAKING COMMIT FETCH
# ============================================================

def fetch_breaking_commit(
    repo_dir,
    case,
    breaking_commit
):
    """
    Make sure the exact BUMP breaking commit exists locally.

    First try the PR head.

    If that fails, try fetching the exact commit.
    """

    pull_number = get_pull_number(case)

    if pull_number is not None:

        print(
            f"  Fetching PR #{pull_number}...",
            flush=True
        )

        local_ref = (
            f"refs/remotes/origin/"
            f"bump-pr-{pull_number}"
        )

        result = run_git(
            [
                "fetch",
                "origin",
                f"refs/pull/{pull_number}/head:"
                f"{local_ref}",
            ],
            repo_dir
        )

        if result and result.returncode == 0:

            check = run_git(
                [
                    "cat-file",
                    "-e",
                    f"{breaking_commit}^{{commit}}",
                ],
                repo_dir
            )

            if check and check.returncode == 0:

                print(
                    "  BUMP breaking commit is available.",
                    flush=True
                )

                return True

        elif result and result.stderr:

            print(
                "  PR fetch failed:",
                result.stderr.strip()[:300],
                flush=True
            )

    print(
        "  Trying exact commit fetch...",
        flush=True
    )

    result = run_git(
        [
            "fetch",
            "origin",
            breaking_commit,
        ],
        repo_dir
    )

    if result and result.returncode == 0:

        check = run_git(
            [
                "cat-file",
                "-e",
                f"{breaking_commit}^{{commit}}",
            ],
            repo_dir
        )

        if check and check.returncode == 0:

            print(
                "  Exact breaking commit is available.",
                flush=True
            )

            return True

    print(
        "  Could not fetch the breaking commit.",
        flush=True
    )

    return False


# ============================================================
# PARENT COMMIT
# ============================================================

def get_parent_commit(repo_dir, commit):

    result = run_git(
        [
            "rev-list",
            "--parents",
            "-n",
            "1",
            commit,
        ],
        repo_dir
    )

    if not result or result.returncode != 0:
        return None

    parts = result.stdout.strip().split()

    if len(parts) >= 2:
        return parts[1]

    return None


# ============================================================
# MAVEN COMMAND
# ============================================================

def get_maven_command(repo_dir):
    """
    Return Maven as an argument list instead of a shell command string.
    This avoids cmd.exe -> mvn.cmd indirection on Windows.
    """

    mvnw = repo_dir / "mvnw.cmd"

    if mvnw.exists():
        return [
            str(mvnw),
            "test",
            "-B",
        ]

    return [
        MAVEN,
        "test",
        "-B",
    ]


# ============================================================
# CHECKOUT
# ============================================================

def checkout_commit(repo_dir, commit):

    result = run_git(
        [
            "checkout",
            "--force",
            commit,
        ],
        repo_dir
    )

    if not result or result.returncode != 0:

        if result and result.stderr:

            print(
                "  Checkout failed:",
                result.stderr.strip()[:300],
                flush=True
            )

        return False

    return True


def has_real_tests(log_path):
    """Return True only when Maven reports at least one executed test.

    A BUILD SUCCESS with only "No tests to run." is not sufficient
    evidence for a SAFE sample.
    """
    if not log_path:
        return False

    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except (OSError, TypeError):
        return False

    if re.search(r"No tests to run\.", text, re.IGNORECASE):
        # A multi-module build can contain one module with no tests and
        # another module with executed tests. Only reject when there is
        # no positive test-count evidence anywhere in the log.
        pass

    # Surefire/Failsafe commonly reports: Tests run: N, Failures: 0, ...
    # Require N >= 1.
    return bool(re.search(r"Tests run:\s*[1-9]\d*", text, re.IGNORECASE))


def normalize_existing_results(results):
    """Invalidate previously recorded SAFE rows that have no executed tests.

    This prevents earlier PASS->PASS cases such as projects with
    "No tests to run" from remaining in the accumulated dataset after
    the stricter SAFE rule is introduced.
    """
    changed = 0

    for row in results:
        if row.get("classification") != "SAFE":
            continue

        pre_log = row.get("pre_log")
        post_log = row.get("post_log")

        if not has_real_tests(pre_log) or not has_real_tests(post_log):
            row["classification"] = "INVALID"
            changed += 1

    if changed:
        print(
            f"  Reclassified {changed} previous SAFE result(s) as INVALID "
            "because no executed tests were detected.",
            flush=True,
        )

    return results


# ============================================================
# RESULT CREATION
# ============================================================

def create_result(**kwargs):

    row = {
        field: None
        for field in RESULT_FIELDS
    }

    row.update(kwargs)

    return row


# ============================================================
# VERIFY ONE CASE
# ============================================================

def verify_case(json_file):

    print(
        "\n" + "=" * 80,
        flush=True
    )

    print(
        "VERIFYING:",
        json_file.name,
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )

    # --------------------------------------------------------
    # READ JSON
    # --------------------------------------------------------

    try:

        case = json.loads(
            json_file.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        print(
            "Could not read JSON:",
            exc,
            flush=True
        )

        return None

    # --------------------------------------------------------
    # EXTRACT METADATA
    # --------------------------------------------------------

    project = case.get("project")

    breaking_commit = case.get(
        "breakingCommit"
    )

    dependency = case.get(
        "updatedDependency",
        {}
    )

    group_id = dependency.get(
        "dependencyGroupID"
    )

    artifact_id = dependency.get(
        "dependencyArtifactID"
    )

    old_version = dependency.get(
        "previousVersion"
    )

    new_version = dependency.get(
        "newVersion"
    )

    java_version = case.get(
        "javaVersionUsedForReproduction"
    )

    failure_category = case.get(
        "failureCategory"
    )

    repo_url = get_repo_url(case)

    print(
        f"Project      : {project}",
        flush=True
    )

    print(
        f"Repository   : {repo_url}",
        flush=True
    )

    print(
        f"Dependency   : {group_id}:{artifact_id}",
        flush=True
    )

    print(
        f"Version      : {old_version} -> {new_version}",
        flush=True
    )

    print(
        f"Java version : {java_version}",
        flush=True
    )

    print(
        f"Breaking     : {breaking_commit}",
        flush=True
    )

    # --------------------------------------------------------
    # BASE RESULT
    # --------------------------------------------------------

    base = dict(
        project=project,
        repo_url=repo_url or "",
        dependency_group_id=group_id,
        dependency_artifact_id=artifact_id,
        old_version=old_version,
        new_version=new_version,
        java_version=java_version,
        failure_category=failure_category,
        breaking_commit=breaking_commit,
        source_json=json_file.name,
    )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if not repo_url or not breaking_commit:

        print(
            "  Missing repository or breaking commit - INVALID",
            flush=True
        )

        return create_result(
            **base,
            pre_update_status="ERROR",
            post_update_status="ERROR",
            classification="INVALID",
        )

    # --------------------------------------------------------
    # PREPARE WORK DIRECTORY
    # --------------------------------------------------------

    repo_dir = WORK_DIR / project

    print(
        "\nRemoving previous repository...",
        flush=True
    )

    if not remove_repository(repo_dir):

        return create_result(
            **base,
            pre_update_status="ERROR",
            post_update_status="ERROR",
            classification="INVALID",
        )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CLONE
    # --------------------------------------------------------

    print(
        "\n" + "-" * 60,
        flush=True
    )

    print(
        "CLONING REPOSITORY",
        flush=True
    )

    print(
        "-" * 60,
        flush=True
    )

    clone_env = os.environ.copy()

    clone_env["GIT_TERMINAL_PROMPT"] = "0"

    try:

        clone = subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                repo_url,
                str(repo_dir),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=clone_env,
            timeout=GIT_CLONE_TIMEOUT_SECONDS,
        )

    except subprocess.TimeoutExpired:

        print(
            f"  Git clone timed out after "
            f"{GIT_CLONE_TIMEOUT_SECONDS}s - INVALID",
            flush=True
        )

        return create_result(
            **base,
            pre_update_status="ERROR",
            post_update_status="ERROR",
            classification="INVALID",
        )

    except Exception as exc:

        print(
            f"  Git clone error: {exc}",
            flush=True
        )

        return create_result(
            **base,
            pre_update_status="ERROR",
            post_update_status="ERROR",
            classification="INVALID",
        )

    if clone.returncode != 0:

        print(
            "  CLONE FAILED:",
            (clone.stderr or "")[:500],
            flush=True
        )

        return create_result(
            **base,
            pre_update_status="ERROR",
            post_update_status="ERROR",
            classification="INVALID",
            pre_exit_code=clone.returncode,
        )

    # --------------------------------------------------------
    # FETCH BREAKING COMMIT
    # --------------------------------------------------------

    print(
        "\n" + "-" * 60,
        flush=True
    )

    print(
        "FETCHING BREAKING COMMIT",
        flush=True
    )

    print(
        "-" * 60,
        flush=True
    )

    if not fetch_breaking_commit(
        repo_dir,
        case,
        breaking_commit
    ):

        return create_result(
            **base,
            pre_update_status="ERROR",
            post_update_status="ERROR",
            classification="INVALID",
        )

    # --------------------------------------------------------
    # FIND PARENT
    # --------------------------------------------------------

    parent_commit = get_parent_commit(
        repo_dir,
        breaking_commit
    )

    if not parent_commit:

        print(
            "  Could not resolve parent commit - INVALID",
            flush=True
        )

        return create_result(
            **base,
            pre_update_status="ERROR",
            post_update_status="ERROR",
            classification="INVALID",
        )

    base["parent_commit"] = parent_commit

    print(
        f"  Parent commit: {parent_commit}",
        flush=True
    )

    # --------------------------------------------------------
    # CHECKOUT PARENT
    # --------------------------------------------------------

    print(
        "\n" + "-" * 60,
        flush=True
    )

    print(
        "CHECKOUT PARENT",
        flush=True
    )

    print(
        "-" * 60,
        flush=True
    )

    if not checkout_commit(
        repo_dir,
        parent_commit
    ):

        return create_result(
            **base,
            pre_update_status="ERROR",
            post_update_status="ERROR",
            classification="INVALID",
        )

    # --------------------------------------------------------
    # MAVEN SETUP
    # --------------------------------------------------------

    maven_command = get_maven_command(
        repo_dir
    )

    print(
        f"  Maven command: {maven_command}",
        flush=True
    )

    log_stub = (
        f"{project}_{breaking_commit[:8]}"
    )

    pre_log = (
        LOG_DIR /
        f"{log_stub}_pre.log"
    )

    post_log = (
        LOG_DIR /
        f"{log_stub}_post.log"
    )

    # --------------------------------------------------------
    # PRE-UPDATE BUILD
    # --------------------------------------------------------

    print(
        "\n" + "-" * 60,
        flush=True
    )

    print(
        "PRE-UPDATE BUILD",
        flush=True
    )

    print(
        "-" * 60,
        flush=True
    )

    pre_status, pre_exit = run_build(
        maven_command,
        repo_dir,
        MAVEN_TIMEOUT_SECONDS,
        pre_log,
    )

    # --------------------------------------------------------
    # CRITICAL TIMEOUT RULE
    #
    # If baseline fails/times out, we cannot establish
    # that the update caused the failure.
    #
    # Therefore:
    #
    # TIMEOUT -> INVALID
    # and DO NOT RUN POST.
    # --------------------------------------------------------

    if pre_status == "TIMEOUT":

        print(
            "  PRE-UPDATE TIMEOUT -> INVALID",
            flush=True
        )

        print(
            "  Skipping post-update build.",
            flush=True
        )

        return create_result(
            **base,
            pre_update_status="TIMEOUT",
            post_update_status="NOT_RUN",
            classification="INVALID",
            pre_exit_code=None,
            post_exit_code=None,
            pre_log=str(pre_log),
            post_log="",
        )

    # --------------------------------------------------------
    # OTHER PRE BUILD ERRORS
    # --------------------------------------------------------

    if pre_status == "ERROR":

        print(
            "  PRE-UPDATE BUILD ERROR -> INVALID",
            flush=True
        )

        return create_result(
            **base,
            pre_update_status="ERROR",
            post_update_status="NOT_RUN",
            classification="INVALID",
            pre_exit_code=pre_exit,
            post_exit_code=None,
            pre_log=str(pre_log),
            post_log="",
        )

    # --------------------------------------------------------
    # CHECKOUT BREAKING COMMIT
    # --------------------------------------------------------

    print(
        "\n" + "-" * 60,
        flush=True
    )

    print(
        "CHECKOUT BREAKING COMMIT",
        flush=True
    )

    print(
        "-" * 60,
        flush=True
    )

    if not checkout_commit(
        repo_dir,
        breaking_commit
    ):

        return create_result(
            **base,
            pre_update_status=pre_status,
            post_update_status="ERROR",
            classification="INVALID",
            pre_exit_code=pre_exit,
            post_exit_code=None,
            pre_log=str(pre_log),
            post_log="",
        )

    # --------------------------------------------------------
    # POST-UPDATE BUILD
    # --------------------------------------------------------

    print(
        "\n" + "-" * 60,
        flush=True
    )

    print(
        "POST-UPDATE BUILD",
        flush=True
    )

    print(
        "-" * 60,
        flush=True
    )

    post_status, post_exit = run_build(
        maven_command,
        repo_dir,
        MAVEN_TIMEOUT_SECONDS,
        post_log,
    )

    # --------------------------------------------------------
    # POST TIMEOUT / ERROR
    # --------------------------------------------------------

    if post_status == "TIMEOUT":

        print(
            "  POST-UPDATE TIMEOUT -> INVALID",
            flush=True
        )

        return create_result(
            **base,
            pre_update_status=pre_status,
            post_update_status="TIMEOUT",
            classification="INVALID",
            pre_exit_code=pre_exit,
            post_exit_code=None,
            pre_log=str(pre_log),
            post_log=str(post_log),
        )

    if post_status == "ERROR":

        print(
            "  POST-UPDATE BUILD ERROR -> INVALID",
            flush=True
        )

        return create_result(
            **base,
            pre_update_status=pre_status,
            post_update_status="ERROR",
            classification="INVALID",
            pre_exit_code=pre_exit,
            post_exit_code=post_exit,
            pre_log=str(pre_log),
            post_log=str(post_log),
        )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    classification = "INVALID"

    if (
        pre_status == "PASS"
        and post_status == "FAIL"
    ):

        classification = "BREAKING"

    elif (
        pre_status == "PASS"
        and post_status == "PASS"
    ):

        if has_real_tests(pre_log) and has_real_tests(post_log):
            classification = "SAFE"
        else:
            print(
                "  PASS->PASS but no executed tests detected -> INVALID",
                flush=True,
            )
            classification = "INVALID"

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print(
        "\nRESULT:",
        f"pre={pre_status}",
        f"post={post_status}",
        f"-> {classification}",
        flush=True
    )

    return create_result(
        **base,
        pre_update_status=pre_status,
        post_update_status=post_status,
        classification=classification,
        pre_exit_code=pre_exit,
        post_exit_code=post_exit,
        pre_log=str(pre_log),
        post_log=str(post_log),
    )


# ============================================================
# RESULT I/O
# ============================================================

def load_existing_results():

    if not OUTPUT_JSON.exists():
        return []

    try:

        data = json.loads(
            OUTPUT_JSON.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


def save_results(results):

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=RESULT_FIELDS
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)


# ============================================================
# SOURCE FILES
# ============================================================

def get_source_files(source_key):

    source_dir = SOURCE_DIRS[source_key]

    if not source_dir.exists():

        print(
            "Source directory not found:",
            source_dir
        )

        return []

    return sorted(
        source_dir.glob("*.json")
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    total = len(results)

    breaking = sum(
        1
        for row in results
        if row.get("classification") == "BREAKING"
    )

    safe = sum(
        1
        for row in results
        if row.get("classification") == "SAFE"
    )

    invalid = sum(
        1
        for row in results
        if row.get("classification") == "INVALID"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "TASK 2 VERIFICATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "TOTAL   :",
        total
    )

    print(
        "BREAKING:",
        breaking
    )

    print(
        "SAFE    :",
        safe
    )

    print(
        "INVALID :",
        invalid
    )

    print(
        "=" * 80
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python extract_verify.py <json-file>"
        )
        print(
            "  python extract_verify.py "
            "--limit N "
            "[--source benchmark|unsuccessful]"
        )
        print(
            "  python extract_verify.py "
            "--all "
            "[--source benchmark|unsuccessful]"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    source_key = "benchmark"

    if "--source" in sys.argv:

        idx = sys.argv.index(
            "--source"
        )

        if idx + 1 >= len(sys.argv):

            print(
                "Missing value for --source"
            )

            sys.exit(1)

        source_key = sys.argv[
            idx + 1
        ]

        del sys.argv[
            idx:idx + 2
        ]

        if source_key not in SOURCE_DIRS:

            print(
                "Unknown --source:",
                source_key
            )

            print(
                "Use: benchmark or unsuccessful"
            )

            sys.exit(1)

    # --------------------------------------------------------
    # LOAD PREVIOUS RESULTS
    # --------------------------------------------------------

    results = load_existing_results()

    # Apply the current SAFE validation rule to previously saved results.
    results = normalize_existing_results(results)

    processed = {
        row.get("source_json")
        for row in results
        if row.get("source_json")
    }

    # --------------------------------------------------------
    # DETERMINE FILES
    # --------------------------------------------------------

    first_arg = sys.argv[1]

    # Single JSON file
    if not first_arg.startswith("--"):

        json_file = Path(
            first_arg
        )

        if not json_file.exists():

            print(
                "File not found:",
                json_file
            )

            sys.exit(1)

        files_to_process = [
            json_file
        ]

    # Limited batch
    elif first_arg == "--limit":

        if len(sys.argv) < 3:

            print(
                "Missing number after --limit"
            )

            sys.exit(1)

        try:

            limit = int(
                sys.argv[2]
            )

        except ValueError:

            print(
                "--limit must be an integer"
            )

            sys.exit(1)

        files_to_process = [
            file
            for file in get_source_files(
                source_key
            )
            if file.name not in processed
        ][:limit]

    # Entire source
    elif first_arg == "--all":

        files_to_process = [
            file
            for file in get_source_files(
                source_key
            )
            if file.name not in processed
        ]

    else:

        print(
            "Unknown argument:",
            first_arg
        )

        sys.exit(1)

    # --------------------------------------------------------
    # NOTHING TO DO
    # --------------------------------------------------------

    if not files_to_process:

        print(
            "No new files to process."
        )

        print_summary(results)

        return

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "TASK 2 DATASET VERIFICATION"
    )

    print(
        "=" * 80
    )

    print(
        "Source:",
        source_key
    )

    print(
        "Files to process:",
        len(files_to_process)
    )

    print(
        "Previously processed:",
        len(processed)
    )

    print(
        "Maven timeout:",
        MAVEN_TIMEOUT_SECONDS,
        "seconds"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # PROCESS CASES
    # --------------------------------------------------------

    for index, json_file in enumerate(
        files_to_process,
        start=1
    ):

        print(
            f"\nCASE {index}/{len(files_to_process)}",
            flush=True
        )

        try:

            result = verify_case(
                json_file
            )

        except KeyboardInterrupt:

            print(
                "\nInterrupted by user.",
                flush=True
            )

            print(
                "Already completed cases "
                "have been saved.",
                flush=True
            )

            break

        except Exception as exc:

            print(
                f"  UNEXPECTED ERROR on "
                f"{json_file.name}: {exc}",
                flush=True
            )

            print(
                "  Marking INVALID and continuing.",
                flush=True
            )

            result = create_result(
                source_json=json_file.name,
                pre_update_status="ERROR",
                post_update_status="ERROR",
                classification="INVALID",
            )

        if result is not None:

            results.append(result)

            # Save after EVERY case.
            save_results(results)

            print(
                "  Result saved.",
                flush=True
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_summary(results)

    print(
        "\nCSV:",
        OUTPUT_CSV
    )

    print(
        "JSON:",
        OUTPUT_JSON
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()