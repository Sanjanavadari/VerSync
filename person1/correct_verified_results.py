import csv
from pathlib import Path


INPUT_FILE = Path("person1") / "verified_breaking_samples.csv"


def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        return

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    for row in rows:

        # ---------------------------------------------------------
        # Corrected graphql-jpa-query verification
        # Independently re-verified:
        # PASS -> PASS = SAFE
        # ---------------------------------------------------------
        if row["source_json"] == "164ee160adc0f663c2069bdbfcdd60e8596327b8.json":

            row["pre_update_status"] = "PASS"
            row["post_update_status"] = "PASS"
            row["classification"] = "SAFE"

        # ---------------------------------------------------------
        # PGS was independently re-verified:
        # FAIL -> FAIL = INVALID
        # ---------------------------------------------------------
        elif row["source_json"] == "17f2bcaaba4805b218743f575919360c5aec5da4.json":

            row["pre_update_status"] = "FAIL"
            row["post_update_status"] = "FAIL"
            row["classification"] = "INVALID"

    fieldnames = list(rows[0].keys())

    with open(
        INPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("VERIFIED RESULTS CORRECTED")
    print("=" * 60)

    print(
        "BREAKING:",
        sum(1 for r in rows if r["classification"] == "BREAKING")
    )

    print(
        "SAFE:",
        sum(1 for r in rows if r["classification"] == "SAFE")
    )

    print(
        "INVALID:",
        sum(1 for r in rows if r["classification"] == "INVALID")
    )

    print("\nUpdated:")
    print(INPUT_FILE)


if __name__ == "__main__":
    main()