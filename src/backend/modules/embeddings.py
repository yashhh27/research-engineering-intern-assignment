

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve()
# parents: [0]=modules, [1]=backend, [2]=src, [3]=project root
PROJECT_ROOT = HERE.parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

EMBEDDINGS_PARQUET_PATH = ARTIFACTS_DIR / "embeddings.parquet"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
EMBEDDINGS_META_JSON_PATH = ARTIFACTS_DIR / "embeddings_meta.json"

# HuggingFace / SentenceTransformer model for embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Lazily load and cache the SentenceTransformer model.

    Returns
    -------
    SentenceTransformer
        Loaded `all-MiniLM-L6-v2` model instance.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def encode_texts(
    texts: Iterable[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a collection of texts into embeddings.

    Parameters
    ----------
    texts : iterable of str
        Input texts (e.g., post `full_text`) to embed.
    batch_size : int, optional
        Batch size for model inference. Larger batches are faster but use
        more RAM. For your dataset (8.8k posts), 32–128 is a good range.
    normalize : bool, optional
        If True, L2-normalize embeddings so that dot product ≈ cosine
        similarity. This is recommended for FAISS IndexFlatIP.

    Returns
    -------
    np.ndarray
        Array of shape (N, D) where N = number of texts, D = embedding dim.
        dtype is float32.
    """
    model = get_model()
    # SentenceTransformers returns numpy.ndarray directly.
    logger.info("Encoding %d texts with batch_size=%d", len(list(texts)), batch_size)

    # We need texts as a concrete list to compute len() and iterate twice.
    text_list = list(texts)
    if not text_list:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    embeddings = model.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    if normalize and embeddings.shape[0] > 0:
        faiss.normalize_L2(embeddings)

    return embeddings


def build_embeddings_dataframe(
    df: pd.DataFrame,
    text_col: str = "full_text",
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Build a DataFrame of embeddings joined with key metadata columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The cleaned, ML-ready DataFrame from `load_as_dataframe`, expected
        to contain at least a `full_text` column.
    text_col : str, optional
        Which column to embed. Default: 'full_text'.
    batch_size : int, optional
        Batch size for embedding generation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
            - post_id (int)
            - title
            - subreddit
            - timestamp
            - url
            - score
            - comments
            - embedding  (list[float] for each row)
    """
    if text_col not in df.columns:
        raise KeyError(f"DataFrame does not contain text column: {text_col}")

    logger.info("Building embeddings for %d posts using column '%s'", len(df), text_col)

    embeddings = encode_texts(df[text_col].tolist(), batch_size=batch_size)
    dim = embeddings.shape[1] if embeddings.size > 0 else 0
    logger.info("Generated embeddings with dimension D=%d", dim)

    # Create a stable integer ID for each post (use current index)
    post_ids = df.index.to_series().astype(int)

    # Build the embeddings DataFrame with basic metadata
    emb_df = pd.DataFrame(
        {
            "post_id": post_ids.values,
            "title": df.get("title", pd.Series([""] * len(df))).astype(str).values,
            "subreddit": df.get("subreddit", pd.Series([""] * len(df))).astype(str).values,
            "timestamp": df.get("timestamp", pd.Series([pd.NaT] * len(df))).values,
            "url": df.get("url", pd.Series([""] * len(df))).astype(str).values,
            "score": df.get("score", pd.Series([0] * len(df))).astype(int).values,
            "comments": df.get("comments", pd.Series([0] * len(df))).astype(int).values,
        }
    )

    # Store embeddings as Python lists so they serialize easily to parquet/JSON
    emb_df["embedding"] = [vec.tolist() for vec in embeddings]

    return emb_df


# ---------------------------------------------------------------------------
# FAISS index helpers
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index (Inner Product / cosine similarity) from embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (N, D), float32, preferably already L2-normalized.

    Returns
    -------
    faiss.Index
        A FAISS IndexFlatIP index over the given vectors.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (N, D), got shape {embeddings.shape}")

    n, dim = embeddings.shape
    logger.info("Building FAISS IndexFlatIP with N=%d, D=%d", n, dim)

    index = faiss.IndexFlatIP(dim)
    if n > 0:
        index.add(embeddings)

    return index


def save_faiss_index(index: faiss.Index, path: Path = FAISS_INDEX_PATH) -> None:
    """
    Persist FAISS index to disk.

    Parameters
    ----------
    index : faiss.Index
        The FAISS index to save.
    path : Path, optional
        Where to write the binary index file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving FAISS index to: %s", path)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path = FAISS_INDEX_PATH) -> faiss.Index:
    """
    Load FAISS index from disk.

    Parameters
    ----------
    path : Path, optional
        Path to the index file.

    Returns
    -------
    faiss.Index
    """
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {path}")
    logger.info("Loading FAISS index from: %s", path)
    return faiss.read_index(str(path))


# ---------------------------------------------------------------------------
# Persistence helpers for embeddings + metadata
# ---------------------------------------------------------------------------

def save_embeddings_parquet(df: pd.DataFrame, path: Path = EMBEDDINGS_PARQUET_PATH) -> None:
    """
    Save embeddings DataFrame to parquet.

    Parameters
    ----------
    df : pandas.DataFrame
        Embeddings DataFrame from `build_embeddings_dataframe`.
    path : Path, optional
        Destination path under `artifacts/`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving embeddings DataFrame to: %s", path)
    df.to_parquet(path, index=False)


def load_embeddings_parquet(path: Path = EMBEDDINGS_PARQUET_PATH) -> pd.DataFrame:
    """
    Load embeddings DataFrame from parquet.

    Parameters
    ----------
    path : Path, optional
        Path to the parquet file.

    Returns
    -------
    pandas.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"Embeddings parquet not found at: {path}")
    logger.info("Loading embeddings DataFrame from: %s", path)
    return pd.read_parquet(path)


def write_embeddings_metadata(
    n_vectors: int,
    dim: int,
    model_name: str = MODEL_NAME,
) -> None:
    """
    Write a small JSON metadata file describing the embeddings.

    Parameters
    ----------
    n_vectors : int
        Number of embedded posts.
    dim : int
        Embedding dimensionality.
    model_name : str
        HuggingFace model identifier.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "model_name": model_name,
        "embedding_dimension": dim,
        "num_vectors": n_vectors,
    }
    EMBEDDINGS_META_JSON_PATH.write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info("Wrote embeddings metadata to: %s", EMBEDDINGS_META_JSON_PATH)


# ---------------------------------------------------------------------------
# High-level pipeline helper
# ---------------------------------------------------------------------------

def build_and_persist_embeddings(df: pd.DataFrame, text_col: str = "full_text") -> Tuple[pd.DataFrame, faiss.Index]:
    """
    High-level convenience function:
    - Builds embeddings from the given DataFrame.
    - Persists `embeddings.parquet`, `faiss_index.bin`, and `embeddings_meta.json`.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned posts DataFrame from `load_as_dataframe`.
    text_col : str, optional
        Column to embed (default 'full_text').

    Returns
    -------
    (emb_df, index) : tuple
        - emb_df: embeddings DataFrame
        - index:  FAISS index built on those embeddings
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    emb_df = build_embeddings_dataframe(df, text_col=text_col)
    logger.info("Embeddings DataFrame shape: %s", emb_df.shape)

    # Convert embedding column back to numpy for FAISS
    if len(emb_df) == 0:
        # Build an empty index with the correct dimension
        dim = get_model().get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        save_faiss_index(index)
        save_embeddings_parquet(emb_df)
        write_embeddings_metadata(0, dim)
        return emb_df, index

    matrix = np.vstack(emb_df["embedding"].apply(np.array).values).astype("float32")
    # Ensure normalized for cosine similarity
    faiss.normalize_L2(matrix)
    index = build_faiss_index(matrix)

    # Persist artifacts
    save_faiss_index(index)
    save_embeddings_parquet(emb_df)
    dim = matrix.shape[1]
    write_embeddings_metadata(len(emb_df), dim)

    return emb_df, index


__all__ = [
    "get_model",
    "encode_texts",
    "build_embeddings_dataframe",
    "build_faiss_index",
    "save_faiss_index",
    "load_faiss_index",
    "save_embeddings_parquet",
    "load_embeddings_parquet",
    "write_embeddings_metadata",
    "build_and_persist_embeddings",
]
