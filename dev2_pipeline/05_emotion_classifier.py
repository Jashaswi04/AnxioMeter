"""Score the Phase 1 corpus with the multi-label GoEmotions model."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL = "SamLowe/roberta-base-go_emotions"
LABELS = ("fear", "nervousness", "anger", "annoyance", "admiration", "gratitude", "optimism", "sadness")


def score_frame(frame, classifier, batch_size=16):
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Input must have unique, nonempty record IDs")
    texts = frame["text_for_analysis"].fillna("").astype(str).tolist()
    # The model is multi-label: each label has an independent probability.
    predictions = classifier(texts, batch_size=batch_size, truncation=True, max_length=512)
    if len(predictions) != len(frame):
        raise ValueError("Classifier returned the wrong number of predictions")
    output = frame.copy()
    for label in LABELS:
        output[f"p_{label}"] = [
            next((float(item["score"]) for item in scores if item["label"] == label), 0.0)
            for scores in predictions
        ]
    output["emotion_model"] = MODEL
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/topic_mapped_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/emotion_scored.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    from transformers import pipeline

    frame = pd.read_csv(args.input, keep_default_na=False)
    required = {"id", "platform", "topic_id", "topic_name", "text_for_analysis", "is_synthetic"}
    if missing := required - set(frame.columns):
        parser.error(f"Missing columns: {sorted(missing)}")
    classifier = pipeline("text-classification", model=MODEL, top_k=None, device=-1)
    scored = score_frame(frame, classifier, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"Scored {len(scored)} records -> {args.output}")


if __name__ == "__main__":
    main()
