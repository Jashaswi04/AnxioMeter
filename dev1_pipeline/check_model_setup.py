import platform

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "cleaned_corpus.csv"


def main():
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print("BERTopic import successful.")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "cleaned_corpus.csv is missing. Run the cleaner first."
        )

    df = pd.read_csv(INPUT_PATH, keep_default_na=False)

    if "cleaned_text" not in df.columns:
        raise ValueError("The dataset is missing cleaned_text.")

    if len(df) < 3:
        raise ValueError("At least three records are needed for this check.")

    documents = df["cleaned_text"].head(3).tolist()

    if any(not text.strip() for text in documents):
        raise ValueError("Empty text found in the selected records.")

    print("Loading all-MiniLM-L6-v2...")
    print("The first run downloads the model and needs internet access.")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    assert embeddings.shape[0] == len(documents)
    assert embeddings.shape[1] > 0
    assert np.isfinite(embeddings).all()

    print(f"Documents encoded: {len(documents)}")
    print(f"Embedding shape: {embeddings.shape}")
    print("Model setup successful.")


if __name__ == "__main__":
    main()