"""
CLI script to build embeddings + FAISS index for the SimPPL dashboard.

Usage (from project root):
    python -m src.backend.scripts.build_embeddings
"""

from __future__ import annotations

from pathlib import Path

from src.backend.modules.data_loader import load_as_dataframe
from src.backend.modules.embeddings import build_and_persist_embeddings

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "data.jsonl"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    print("=== Building embeddings + FAISS index ===")
    print(f"Loading dataset from: {DATA_PATH}")

    df = load_as_dataframe(DATA_PATH)
    print(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    emb_df, index = build_and_persist_embeddings(df, text_col="full_text")

    print("\n=== Embedding Pipeline Complete ===")
    print(f"Embedded posts: {len(emb_df)}")
    print("Artifacts written under: artifacts/")
    print(" - embeddings.parquet")
    print(" - faiss_index.bin")
    print(" - embeddings_meta.json")


if __name__ == "__main__":
    main()
