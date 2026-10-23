import json
import subprocess
import sys
from pathlib import Path


def run_docker_command(command):
    print("\nRunning:")
    print(command)

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    output = (result.stdout or "") + "\n" + (result.stderr or "")

    print("\nDocker exit code:", result.returncode)

    if result.stdout:
        print("\n--- STDOUT ---")
        print(result.stdout)

    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)

    # ---------------------------------------------------------
    # Detect Maven's final build result.
    #
    # BUMP containers may return Docker exit code 0 even when
    # Maven fails. Therefore, Maven's BUILD result is used.
    #
    # Maven output can contain errors/warnings before the final
    # result, so we collect all explicit BUILD SUCCESS/FAILURE
    # lines and use the LAST one.
    # ---------------------------------------------------------

    build_results = []

    for line in output.splitlines():
        line = line.strip()

        if line == "[INFO] BUILD SUCCESS":
            build_results.append("PASS")

        elif line == "[INFO] BUILD FAILURE":
            build_results.append("FAIL")

    if build_results:
        status = build_results[-1]

    else:
        # If Maven did not print a recognizable BUILD result,
        # we cannot confidently classify the case.
        status = "ERROR"

    print("Detected Maven status:", status)

    return status


def verify_bump_case(json_file):
    json_path = Path(json_file)

    with open(json_path, "r", encoding="utf-8") as f:
        case = json.load(f)

    project = case["project"]

    dependency = case["updatedDependency"]

    group_id = dependency["dependencyGroupID"]
    artifact_id = dependency["dependencyArtifactID"]

    old_version = dependency["previousVersion"]
    new_version = dependency["newVersion"]

    pre_command = case["preCommitReproductionCommand"]
    breaking_command = case["breakingUpdateReproductionCommand"]

    print("=" * 70)
    print("BUMP VERIFICATION")
    print("=" * 70)

    print(f"Project       : {project}")
    print(f"Dependency    : {group_id}:{artifact_id}")
    print(f"Old version   : {old_version}")
    print(f"New version   : {new_version}")
    print(f"Java version  : {case.get('javaVersionUsedForReproduction')}")
    print(f"Expected      : {case.get('failureCategory')}")

    # ---------------------------------------------------------
    # PRE-UPDATE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. PRE-UPDATE VERIFICATION")
    print("=" * 70)

    pre_status = run_docker_command(pre_command)

    # Retry the baseline once if it did not PASS.
    #
    # Some BUMP baseline containers can occasionally produce
    # inconsistent results. We need a valid PASS baseline
    # before calling an update BREAKING or SAFE.
    if pre_status != "PASS":
        print("\nPRE-UPDATE did not PASS.")
        print("Retrying baseline verification...")

        pre_status = run_docker_command(pre_command)

    # ---------------------------------------------------------
    # BREAKING UPDATE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. BREAKING-UPDATE VERIFICATION")
    print("=" * 70)

    breaking_status = run_docker_command(breaking_command)

    # ---------------------------------------------------------
    # CLASSIFICATION
    # ---------------------------------------------------------

    if pre_status == "PASS" and breaking_status == "FAIL":
        classification = "BREAKING"

    elif pre_status == "PASS" and breaking_status == "PASS":
        classification = "SAFE"

    else:
        classification = "INVALID"

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Pre-update status : {pre_status}")
    print(f"Post-update status: {breaking_status}")
    print(f"Classification    : {classification}")

    return {
        "project": project,
        "repo_url": case["url"],
        "dependency_group_id": group_id,
        "dependency_artifact_id": artifact_id,
        "old_version": old_version,
        "new_version": new_version,
        "java_version": case.get("javaVersionUsedForReproduction"),
        "failure_category": case.get("failureCategory"),
        "pre_update_status": pre_status,
        "post_update_status": breaking_status,
        "classification": classification,
        "pre_exit_code": None,
        "post_exit_code": None,
        "source_json": json_path.name
    }


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python bump_verify.py <path-to-bump-json>")
        sys.exit(1)

    result = verify_bump_case(sys.argv[1])

    print("\nStructured result:")
    print(json.dumps(result, indent=2))