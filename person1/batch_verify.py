import json
import csv
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

    # BUMP Docker containers can sometimes return exit code 0
    # even when Maven reports BUILD FAILURE.
    if "BUILD FAILURE" in output:
        status = "FAIL"
    elif "BUILD SUCCESS" in output:
        status = "PASS"
    else:
        status = "ERROR"

    print("Docker exit code:", result.returncode)
    print("Detected Maven status:", status)

    return status


def verify_case(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        case = json.load(f)

    dependency = case["updatedDependency"]

    project = case["project"]

    group_id = dependency["dependencyGroupID"]
    artifact_id = dependency["dependencyArtifactID"]

    old_version = dependency["previousVersion"]
    new_version = dependency["newVersion"]

    print("\n" + "=" * 70)
    print(f"PROJECT: {project}")
    print(f"DEPENDENCY: {group_id}:{artifact_id}")
    print(f"VERSION: {old_version} -> {new_version}")
    print("=" * 70)

    print("\n[1/2] PRE-UPDATE")

    pre_status = run_docker_command(
    case["preCommitReproductionCommand"]
    )

    # Retry the baseline once if it did not pass.
    # Some BUMP containers can occasionally produce
    # inconsistent baseline results.
    if pre_status != "PASS":
        print("\nPRE-UPDATE did not PASS.")
        print("Retrying baseline verification...")

        pre_status = run_docker_command(
            case["preCommitReproductionCommand"]
        )

    print("\n[2/2] BREAKING UPDATE")

    post_status = run_docker_command(
        case["breakingUpdateReproductionCommand"]
    )

    if pre_status == "PASS" and post_status == "FAIL":
        classification = "BREAKING"
    elif pre_status == "PASS" and post_status == "PASS":
        classification = "SAFE"
    else:
        classification = "INVALID"

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
        "post_update_status": post_status,
        "classification": classification,
        "source_json": Path(json_file).name
    }


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python batch_verify.py <json-folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])

    # ---------------------------------------------------------
    # STEP 1: Find all BUMP benchmark JSON files
    # ---------------------------------------------------------

    all_json_files = sorted(folder.glob("*.json"))

    print("=" * 70)
    print("BUMP BATCH VERIFICATION")
    print("=" * 70)
    print("Total BUMP JSON files found:", len(all_json_files))

    # ---------------------------------------------------------
    # STEP 2: Select diverse cases
    #
    # Prefer:
    #   - different projects
    #   - different dependencies
    #
    # Maximum selected cases = 50
    # ---------------------------------------------------------

    selected_files = []

    seen_projects = set()
    seen_dependencies = set()

    for json_file in all_json_files:

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                case = json.load(f)

            project = case.get("project")

            dependency = case.get("updatedDependency", {})

            group_id = dependency.get("dependencyGroupID")
            artifact_id = dependency.get("dependencyArtifactID")

            dependency_key = f"{group_id}:{artifact_id}"
            # QuickFIX/J was independently re-verified successfully,
            # so it can be included normally in the final dataset.

            # Prefer a completely new project and dependency.
            if (
                project not in seen_projects
                and dependency_key not in seen_dependencies
            ):
                selected_files.append(json_file)

                seen_projects.add(project)
                seen_dependencies.add(dependency_key)

            # Stop once we have 50 cases.
            if len(selected_files) >= 50:
                break

        except Exception as e:
            print(f"Skipping {json_file.name}: {e}")

    # ---------------------------------------------------------
    # STEP 3: Print selected cases
    #
    # IMPORTANT:
    # We stop here for now.
    # No Docker/Maven builds are executed.
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SELECTED DIVERSE CASES")
    print("=" * 70)

    for index, json_file in enumerate(selected_files, start=1):

        with open(json_file, "r", encoding="utf-8") as f:
            case = json.load(f)

        dependency = case["updatedDependency"]

        print(
            f"{index}. "
            f"{case['project']} | "
            f"{dependency['dependencyGroupID']}:"
            f"{dependency['dependencyArtifactID']} | "
            f"{dependency['previousVersion']} -> "
            f"{dependency['newVersion']}"
        )

    print("\nTotal selected:", len(selected_files))




    # =========================================================
    # VERIFICATION SECTION
    #
    # This will be used after we approve the selected cases.
    # =========================================================

    results = []

    for index, json_file in enumerate(selected_files, start=1):

        print("\n")
        print("#" * 70)
        print(f"CASE {index}/{len(selected_files)}")
        print("#" * 70)

        try:
            result = verify_case(json_file)

            results.append(result)

            print("\nRESULT:")
            print(json.dumps(result, indent=2))

        except Exception as e:
            print(f"ERROR processing {json_file.name}: {e}")

    # ---------------------------------------------------------
    # STEP 5: Batch summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)

    print("Total cases:", len(results))

    breaking = sum(
        1 for r in results
        if r["classification"] == "BREAKING"
    )

    safe = sum(
        1 for r in results
        if r["classification"] == "SAFE"
    )

    invalid = sum(
        1 for r in results
        if r["classification"] == "INVALID"
    )

    print("BREAKING:", breaking)
    print("SAFE:", safe)
    print("INVALID:", invalid)

    # ---------------------------------------------------------
    # STEP 6: Save CSV
    # ---------------------------------------------------------

    output_file = Path("person1") / "verified_breaking_samples.csv"

    fieldnames = [
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
        "source_json"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to:")
    print(output_file)