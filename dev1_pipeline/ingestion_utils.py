import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_COLUMNS = [
    "id",
    "platform",
    "created_utc",
    "text",
    "record_type",
    "parent_id",
    "is_synthetic",
    "source",
]


def ingest_sample(platform):
    if platform not in {"reddit", "linkedin"}:
        raise ValueError(f"Unsupported platform: {platform}")

    source_path = (
        PROJECT_ROOT / "data" / "source_samples"
        / f"{platform}_sample.json"
    )
    output_path = PROJECT_ROOT / "data" / f"raw_{platform}.csv"
    report_path = (
        PROJECT_ROOT / "reports" / f"{platform}_ingestion.json"
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path.name}. "
            "Run 00_create_sample_sources.py first."
        )

    # Prevent an accidental overwrite of an existing dataset.
    if output_path.exists():
        raise FileExistsError(
            f"{output_path.name} already exists. "
            "Inspect and move or remove it before rerunning."
        )

    records = json.loads(source_path.read_text(encoding="utf-8"))

    if not isinstance(records, list) or not records:
        raise ValueError("Source must be a nonempty JSON list.")

    rows = []
    rejected = []
    seen_ids = set()
    duplicate_count = 0

    for position, record in enumerate(records, start=1):
        try:
            if not isinstance(record, dict):
                raise ValueError("Record must be a JSON object.")

            source_id = record.get("source_id")
            text = record.get("content")
            record_type = record.get("record_type")

            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError("Missing or invalid source_id.")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Missing or empty content.")

            if record_type != "post":
                raise ValueError("This sample importer expects posts.")

            if record.get("is_synthetic") is not True:
                raise ValueError("Sample records must be marked synthetic.")

            timestamp = pd.to_datetime(
                record.get("published_at"),
                utc=True,
                errors="raise",
            )

            if pd.isna(timestamp):
                raise ValueError("Missing publication timestamp.")

            record_id = (
                f"{platform}_{record_type}_{source_id.strip()}"
            )

            if record_id in seen_ids:
                duplicate_count += 1
                continue

            seen_ids.add(record_id)

            rows.append({
                "id": record_id,
                "platform": platform,
                "created_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "text": text.strip(),
                "record_type": record_type,
                "parent_id": "",
                "is_synthetic": True,
                "source": source_path.name,
            })

        except (ValueError, TypeError, OverflowError) as error:
            rejected.append({
                "record_number": position,
                "reason": str(error),
            })

    if not rows:
        raise ValueError("No valid records remain after validation.")

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    report = {
        "platform": platform,
        "mode": "synthetic_file_import",
        "input_records": len(records),
        "output_records": len(df),
        "duplicate_records": duplicate_count,
        "rejected_records": len(rejected),
        "rejections": rejected,
        "output_file": output_path.name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8")
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Platform: {platform}")
    print("Mode: synthetic file import")
    print(f"Input records: {len(records)}")
    print(f"Saved records: {len(df)}")
    print(f"Duplicates skipped: {duplicate_count}")
    print(f"Invalid records rejected: {len(rejected)}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")