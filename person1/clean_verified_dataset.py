import csv
from pathlib import Path


INPUT_FILE = Path("person1") / "verified_breaking_samples.csv"

BREAKING_FILE = Path("person1") / "verified_breaking_samples_clean.csv"
SAFE_FILE = Path("person1") / "verified_safe_samples.csv"
INVALID_FILE = Path("person1") / "invalid_samples.csv"


def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    breaking_rows = [
        row for row in rows
        if row["classification"] == "BREAKING"
        and row["pre_update_status"] == "PASS"
        and row["post_update_status"] == "FAIL"
    ]

    safe_rows = [
        row for row in rows
        if row["classification"] == "SAFE"
        and row["pre_update_status"] == "PASS"
        and row["post_update_status"] == "PASS"
    ]

    invalid_rows = [
        row for row in rows
        if row["classification"] == "INVALID"
    ]

    fieldnames = list(rows[0].keys()) if rows else []

    def write_csv(path, data):
        with open(
            path,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(data)

    write_csv(BREAKING_FILE, breaking_rows)
    write_csv(SAFE_FILE, safe_rows)
    write_csv(INVALID_FILE, invalid_rows)

    print("=" * 60)
    print("VERIFIED DATASET CLEANUP")
    print("=" * 60)

    print(f"Total verification results : {len(rows)}")
    print(f"Verified BREAKING samples  : {len(breaking_rows)}")
    print(f"Verified SAFE samples      : {len(safe_rows)}")
    print(f"INVALID samples            : {len(invalid_rows)}")

    print("\nFiles created:")

    print(f"BREAKING : {BREAKING_FILE}")
    print(f"SAFE     : {SAFE_FILE}")
    print(f"INVALID  : {INVALID_FILE}")


if __name__ == "__main__":
    main()