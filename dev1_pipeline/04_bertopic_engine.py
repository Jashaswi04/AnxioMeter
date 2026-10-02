import argparse
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
)
from umap import UMAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Keep verified releases safe from accidental overwriting.
    if output_dir.exists():
        raise FileExistsError(
            "Output directory already exists. Choose a new directory."
        )

    df = pd.read_csv(input_path, keep_default_na=False)

    required = {
        "id",
        "platform",
        "cleaned_text",
        "text_for_analysis",
        "is_synthetic",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if len(df) < 60:
        raise ValueError(
            "Too few records for this configured demonstration."
        )

    if not df["id"].is_unique:
        raise ValueError("Duplicate record IDs found.")

    if df["cleaned_text"].str.strip().eq("").any():
        raise ValueError("Empty cleaned text found.")

    df = df.reset_index(drop=True)
    documents = df["cleaned_text"].tolist()

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    embeddings = embedding_model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=30,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        ),
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = model.fit_transform(
        documents,
        embeddings=embeddings,
    )
    df["topic_id"] = topics

    keyword_stopwords = sorted(
        set(ENGLISH_STOP_WORDS)
        | {
            "service_company",
            "product_company",
            "labor_practice",
        }
    )

    model.update_topics(
        documents,
        vectorizer_model=CountVectorizer(
            stop_words=keyword_stopwords,
            ngram_range=(1, 2),
            min_df=2,
        ),
        ctfidf_model=ClassTfidfTransformer(
            reduce_frequent_words=True
        ),
    )

    output_dir.mkdir(parents=True)

    # These assignments require review before becoming a new release.
    df.to_csv(
        output_dir / "topic_assignments_unreviewed.csv",
        index=False,
    )

    model.get_topic_info().to_csv(
        output_dir / "topic_summary.csv",
        index=False,
    )

    review_rows = []

    for topic_id in sorted(df["topic_id"].unique()):
        examples = df[df["topic_id"] == topic_id].sample(
            n=min(10, int(df["topic_id"].eq(topic_id).sum())),
            random_state=42,
        )

        review_rows.extend(
            examples[
                ["id", "platform", "topic_id", "text_for_analysis"]
            ].to_dict("records")
        )

    pd.DataFrame(review_rows).to_csv(
        output_dir / "topic_review_examples.csv",
        index=False,
    )

    model.save(
        str(output_dir / "bertopic_model"),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    print("\nTraining and export complete.")
    print(f"Records: {len(df)}")
    print(f"Outliers: {int(df['topic_id'].eq(-1).sum())}")
    print(f"Output: {output_dir}")
    print("Review topics before assigning names or publishing a new release.")


if __name__ == "__main__":
    main()