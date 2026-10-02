import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "source_samples"

# Invented examples for testing. These are not actual platform posts.
SAMPLES = {
    "reddit": [
        "I work at TCS with a 12lpa package. I am worried about layoffs.",
        "Infosys put me on the bench. I am not optimistic about my role.",
        "My 90-day notice period is making it difficult to switch jobs.",
        "I received a PIP at Wipro and feel anxious about my future.",
        "My CTC is 8 LPA and this year's appraisal gave me no hike.",
        "Our return-to-office mandate means moving away from my family.",
        "I joined a product startup but constantly feel underqualified.",
        "The deployment pipeline failed, but my team helped me fix it.",
        "I got a MAANG offer and feel happy about the opportunity.",
        "I spent 90 days learning Python and enjoyed the experience.",
    ],
    "linkedin": [
        "Thrilled to share that I joined TCS with a 12 LPA package!",
        "Humbled by this milestone and grateful for my learning journey.",
        "Our team embraces synergy to unlock potential during change.",
        "After a layoff, I am looking for engineering roles. #OpenToWork",
        "Excited to begin a new role at a product startup.",
        "Returning to the office has been difficult due to my commute.",
        "Grateful for my appraisal and the support of my team at HCL.",
        "My 90-day notice period ends soon. I look forward to my new role.",
        "I am not thrilled about the hiring freeze. It creates uncertainty.",
        "We are hiring at Wipro. Compensation starts at 12 lacs annually.",
    ],
}


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    for platform, texts in SAMPLES.items():
        records = []

        for number, text in enumerate(texts, start=1):
            records.append({
                "source_id": f"demo_{platform}_{number:03d}",
                "published_at": f"2026-09-{number:02d}T10:00:00Z",
                "content": text,
                "record_type": "post",
                "is_synthetic": True,
            })

        path = SOURCE_DIR / f"{platform}_sample.json"
        path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Created {path.name}: {len(records)} synthetic records")


if __name__ == "__main__":
    main()