import html
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

# Apply these rules only to cleaned_text.
# Preserve text_for_analysis for emotion and fluff scoring.
NORMALIZATION_RULES = [
    (
        r"\b(?:TCS|Infosys|Wipro|Cognizant|HCL)\b",
        "[SERVICE_COMPANY]",
    ),
    (
        r"\b(?:FAANG|MAANG|product[\s-]+startups?)\b",
        "[PRODUCT_COMPANY]",
    ),
    (
        r"\b\d+(?:\.\d+)?\s*(?:lpa|lakhs?|lacs?)\b",
        "[COMPENSATION]",
    ),
    (
        r"\bCTC\b",
        "[COMPENSATION]",
    ),
    (
        r"\b(?:90[\s-]*days?[\s-]+)?notice[\s-]+period\b",
        "[LABOR_PRACTICE] notice_period",
    ),
    (
        r"\b(?:PIP|performance[\s-]+improvement[\s-]+plan)\b",
        "[LABOR_PRACTICE] performance_improvement_plan",
    ),
    (
        r"\b(?:(?:on|onto)\s+(?:the\s+)?bench|benched)\b",
        "[LABOR_PRACTICE] bench",
    ),
    (
        r"\b(?:bonded[\s-]+(?:employment|labor|labour)"
        r"|employment[\s-]+bond)\b",
        "[LABOR_PRACTICE] employment_bond",
    ),
]

COMPILED_RULES = [
    (re.compile(pattern, flags=re.IGNORECASE), replacement)
    for pattern, replacement in NORMALIZATION_RULES
]


def prepare_analysis_text(text):
    """Light cleaning while preserving sentiment and writing style."""
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[EMAIL]",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_lexicon(text):
    """Normalize domain terms without altering the original column."""
    for pattern, replacement in COMPILED_RULES:
        text = pattern.sub(lambda match: replacement, text)

    return re.sub(r"\s+", " ", text).strip()


def check_cleaning_rules():
    """Check cases where incorrect cleaning would change meaning."""
    assert normalize_lexicon("12lpa") == "[COMPENSATION]"
    assert normalize_lexicon("12 lacs") == "[COMPENSATION]"
    assert normalize_lexicon("TCS") == "[SERVICE_COMPANY]"

    assert normalize_lexicon(
        "The deployment pipeline failed."
    ) == "The deployment pipeline failed."

    assert normalize_lexicon(
        "I spent 90 days learning Python."
    ) == "I spent 90 days learning Python."

    assert normalize_lexicon(
        "My 90-day notice period"
    ) == "My [LABOR_PRACTICE] notice_period"

    assert prepare_analysis_text(
        "I am not thrilled."
    ) == "I am not thrilled."

    assert normalize_lexicon(
        "I sat on a park bench."
    ) == "I sat on a park bench."


def main():
    check_cleaning_rules()

    frames = []
    input_counts = {}

    for platform in ("reddit", "linkedin"):
        path = DATA_DIR / f"raw_{platform}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run its ingestion script first."
            )

        # Keep empty parent_id values as empty strings.
        df = pd.read_csv(path, keep_default_na=False)

        required = {
            "id", "platform", "created_utc", "text",
            "record_type", "parent_id", "is_synthetic", "source",
        }
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        if not df["platform"].eq(platform).all():
            raise ValueError(f"Incorrect platform values in {path.name}")

        input_counts[platform] = len(df)
        frames.append(df)

    corpus = pd.concat(frames, ignore_index=True)

    if corpus.empty:
        raise ValueError("Input files contain no records.")

    if corpus["id"].duplicated().any():
        raise ValueError("Duplicate IDs found. Check ingestion outputs.")

    if corpus["id"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty record IDs found.")

    pd.to_datetime(
        corpus["created_utc"], utc=True, errors="raise"
    )

    corpus["text_for_analysis"] = corpus["text"].map(
        prepare_analysis_text
    )
    corpus["cleaned_text"] = corpus["text_for_analysis"].map(
        normalize_lexicon
    )

    # Remove records that became empty after cleaning.
    empty_mask = corpus["cleaned_text"].eq("")
    excluded_ids = corpus.loc[empty_mask, "id"].tolist()
    corpus = corpus.loc[~empty_mask].copy()

    if corpus.empty:
        raise ValueError("All records became empty after cleaning.")

    changed_count = int(
        corpus["text_for_analysis"].ne(corpus["cleaned_text"]).sum()
    )

    output_path = DATA_DIR / "cleaned_corpus.csv"
    corpus.to_csv(output_path, index=False, encoding="utf-8")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "input_counts": input_counts,
        "output_records": len(corpus),
        "empty_records_removed": len(excluded_ids),
        "excluded_ids": excluded_ids,
        "records_with_lexicon_changes": changed_count,
        "output_counts_by_platform": {
            platform: int(count)
            for platform, count in
            corpus["platform"].value_counts().items()
        },
        "normalization_version": "1.0",
    }

    report_path = REPORT_DIR / "cleaning_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    preview_path = REPORT_DIR / "cleaning_preview.csv"
    corpus[
        ["id", "platform", "text", "text_for_analysis", "cleaned_text"]
    ].to_csv(preview_path, index=False, encoding="utf-8")

    print("Cleaning checks passed.")
    print(f"Input records: {sum(input_counts.values())}")
    print(f"Saved records: {len(corpus)}")
    print(f"Empty records removed: {len(excluded_ids)}")
    print(f"Records with lexicon changes: {changed_count}")
    print("Output: data/cleaned_corpus.csv")
    print("Report: reports/cleaning_report.json")
    print("Preview: reports/cleaning_preview.csv")


if __name__ == "__main__":
    main()