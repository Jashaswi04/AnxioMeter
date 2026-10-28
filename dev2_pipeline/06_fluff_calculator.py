"""Calculate transparent, illustrative CAI and CFS scores (both 0 to 100)."""

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUZZWORDS = (
    "synergy", "pivotal", "humbled", "thrilled", "unlock potential",
    "learning journey", "grateful", "excited to share", "honored", "game changer",
)
HUMBLEBRAGS = (
    "i am excited to share", "i'm excited to share", "humbled by this milestone",
    "honored to announce", "thrilled to announce", "grateful to announce",
)
WORDS = re.compile(r"\b[\w']+\b", re.UNICODE)


def matches(text, phrase):
    return len(re.findall(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, flags=re.I))


def fluff_features(text):
    text = str(text or "")
    words = WORDS.findall(text.lower())
    if not words:
        return {"buzzword_count": 0, "humblebrag_count": 0, "type_token_ratio": 0.0, "cfs": 0.0}
    buzz = sum(matches(text, phrase) for phrase in BUZZWORDS)
    humble = sum(matches(text, phrase) for phrase in HUMBLEBRAGS)
    ttr = len(set(words)) / len(words)
    # Per-100-word frequency is capped so a single short post cannot dominate.
    buzz_component = min(1.0, 100 * buzz / len(words) / 10)
    humble_component = min(1.0, 100 * humble / len(words) / 5)
    repetition_component = 1 - ttr
    cfs = 100 * (0.45 * buzz_component + 0.40 * humble_component + 0.15 * repetition_component)
    return {"buzzword_count": buzz, "humblebrag_count": humble,
            "type_token_ratio": round(ttr, 4), "cfs": round(cfs, 2)}


def calculate(frame):
    required = {"id", "platform", "topic_id", "topic_name", "text_for_analysis", "is_synthetic",
                "p_fear", "p_nervousness", "p_anger", "p_annoyance", "p_sadness"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Input must have unique, nonempty record IDs")
    out = frame.copy()
    for col in ("p_fear", "p_nervousness", "p_anger", "p_annoyance", "p_sadness"):
        out[col] = pd.to_numeric(out[col], errors="raise")
        if (~out[col].between(0, 1)).any():
            raise ValueError(f"{col} must contain probabilities between 0 and 1")
    # A named composite measure, not a diagnosis or calibrated clinical scale.
    out["cai"] = (100 * (0.30 * out.p_fear + 0.25 * out.p_nervousness
                            + 0.20 * out.p_anger + 0.15 * out.p_annoyance
                            + 0.10 * out.p_sadness)).round(2)
    features = pd.DataFrame([fluff_features(t) for t in out.text_for_analysis], index=out.index)
    return pd.concat([out, features], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/emotion_scored.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/final_analytics.csv")
    args = parser.parse_args()
    result = calculate(pd.read_csv(args.input, keep_default_na=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Indexed {len(result)} records -> {args.output}")


if __name__ == "__main__":
    main()
