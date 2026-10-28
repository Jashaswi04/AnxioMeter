"""Summarize CAI/CFS per topic and platform, keeping outliers explicit."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def summarize(frame):
    needed = {"platform", "topic_id", "topic_name", "cai", "cfs"}
    if missing := needed - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return (frame.groupby(["topic_id", "topic_name", "platform"], dropna=False)
            .agg(post_count=("cai", "size"), mean_cai=("cai", "mean"), mean_cfs=("cfs", "mean"))
            .reset_index().round({"mean_cai": 2, "mean_cfs": 2}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/final_analytics.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/divergence_summary.csv")
    args = parser.parse_args()
    summary = summarize(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
