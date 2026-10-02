import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source_samples"

THEMES = {
    "layoffs": [
        "Our company announced layoffs and several engineering roles were cut.",
        "My team was dissolved during restructuring and I lost my job.",
        "The latest workforce reduction has left me uncertain about employment.",
        "After being laid off, I am searching for another software role.",
        "Management announced redundancies despite the team meeting its targets.",
    ],
    "notice_period": [
        "My 90-day notice period conflicts with the new employer's joining date.",
        "The recruiter wants an immediate joiner but my notice period is long.",
        "I requested an early release after resigning but it was rejected.",
        "My next job offer may expire before I finish serving notice.",
        "Negotiating a notice period buyout has delayed my job switch.",
    ],
    "compensation": [
        "My 12lpa salary barely changed after this year's appraisal.",
        "I am comparing the fixed salary and variable pay in two offers.",
        "The annual salary hike was smaller than the rise in living costs.",
        "My CTC looks high but the monthly take-home pay is disappointing.",
        "I negotiated a higher compensation package after reviewing market salaries.",
    ],
    "return_to_office": [
        "The return-to-office mandate requires relocating to another city.",
        "My daily commute has increased since compulsory office attendance began.",
        "I prefer remote work because moving closer to the office is expensive.",
        "The new office attendance policy has disrupted my family arrangements.",
        "Our hybrid work arrangement was replaced with five days in the office.",
    ],
    "impostor_feelings": [
        "I feel underqualified when experienced engineers discuss system design.",
        "Despite positive feedback, I worry that my coding skills are inadequate.",
        "I compare myself with teammates and doubt whether I deserve this role.",
        "Asking basic technical questions makes me worry about appearing incompetent.",
        "I completed the project but still feel my success was mostly luck.",
    ],
    "bench_and_pip": [
        "I am on the bench without a client project and worry about my role.",
        "My manager placed me on a PIP with unclear performance expectations.",
        "Being benched for weeks has made my future at the company uncertain.",
        "The performance improvement plan gives me little time to meet its targets.",
        "I need a project allocation before my bench period becomes a problem.",
    ],
}

CONTEXTS = [
    "I work at TCS.",
    "I recently joined Infosys.",
    "My current employer is Wipro.",
    "I work for Cognizant.",
    "I joined HCL last year.",
    "I work at a product startup.",
    "I have two years of software development experience.",
    "I work on backend services.",
    "I recently moved into a testing role.",
    "I am part of a small engineering team.",
    "I work on cloud infrastructure.",
    "I changed teams earlier this year.",
]

REFLECTIONS = [
    "I am worried and would appreciate practical advice.",
    "I am hopeful that the situation will improve.",
    "I feel frustrated by the lack of clear communication.",
    "I am grateful for colleagues who have supported me.",
    "I am documenting the experience to understand my next steps.",
    "This has been a learning journey, although it has been difficult.",
]


def main():
    # Back up previous demo inputs and outputs before replacing them.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_dir = DATA_DIR / "backups" / timestamp

    existing_files = [
        SOURCE_DIR / "reddit_sample.json",
        SOURCE_DIR / "linkedin_sample.json",
        DATA_DIR / "raw_reddit.csv",
        DATA_DIR / "raw_linkedin.csv",
        DATA_DIR / "cleaned_corpus.csv",
        PROJECT_ROOT / "reports" / "reddit_ingestion.json",
        PROJECT_ROOT / "reports" / "linkedin_ingestion.json",
        PROJECT_ROOT / "reports" / "cleaning_report.json",
        PROJECT_ROOT / "reports" / "cleaning_preview.csv",
    ]

    for path in existing_files:
        if path.exists():
            destination = backup_dir / path.relative_to(PROJECT_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    records = {"reddit": [], "linkedin": []}

    for theme_index, sentences in enumerate(THEMES.values()):
        for sentence_index, sentence in enumerate(sentences):
            for context_index, context in enumerate(CONTEXTS):
                # Each core sentence appears equally on both platforms.
                platform = (
                    "reddit"
                    if (sentence_index + context_index) % 2 == 0
                    else "linkedin"
                )

                reflection = REFLECTIONS[
                    (theme_index + sentence_index + context_index)
                    % len(REFLECTIONS)
                ]

                number = len(records[platform]) + 1
                text = f"{context} {sentence} {reflection}"

                records[platform].append({
                    "source_id": f"expanded_{platform}_{number:04d}",
                    "published_at": (
                        f"2026-09-{((number - 1) % 28) + 1:02d}"
                        "T10:00:00Z"
                    ),
                    "content": text,
                    "record_type": "post",
                    "is_synthetic": True,
                })

    all_texts = [
        row["content"]
        for platform_records in records.values()
        for row in platform_records
    ]

    assert len(all_texts) == 360
    assert len(set(all_texts)) == 360, "Duplicate sample texts found."

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    for platform, platform_records in records.items():
        path = SOURCE_DIR / f"{platform}_sample.json"
        path.write_text(
            json.dumps(platform_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"{platform}: {len(platform_records)} synthetic source records")

    # The old raw files have been backed up.
    # Remove them so the ingestion scripts can create fresh outputs.
    for filename in ("raw_reddit.csv", "raw_linkedin.csv"):
        path = DATA_DIR / filename
        if path.exists():
            path.unlink()

    print(f"Backup location: {backup_dir.relative_to(PROJECT_ROOT)}")
    print("Now rerun both ingestion scripts, then the cleaner.")


if __name__ == "__main__":
    main()