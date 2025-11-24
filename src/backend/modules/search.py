from __future__ import annotations

# ================================================================
# Environment setup (must run before SentenceTransformers loads)
# ================================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================================================================
# Imports
# ================================================================
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ================================================================
# Paths & Configuration
# ================================================================
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

EMBEDDINGS_PARQUET_PATH = ARTIFACTS_DIR / "embeddings.parquet"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
EMBEDDINGS_META_PATH = ARTIFACTS_DIR / "embeddings_meta.json"

# Global caches
_MODEL: Optional[SentenceTransformer] = None
_EMB_TABLE: Optional[pd.DataFrame] = None
_FAISS_INDEX: Optional[faiss.Index] = None
_META: Optional[Dict[str, Any]] = None

# Defaults (overwritten by metadata.json)
_METRIC: str = "ip"
_NORMALIZE: bool = True


# ================================================================
# Metadata Loading
# ================================================================
def _load_metadata() -> Dict[str, Any]:
    global _META, _METRIC, _NORMALIZE

    if _META is not None:
        return _META

    if not EMBEDDINGS_META_PATH.exists():
        raise FileNotFoundError(f"Metadata file missing: {EMBEDDINGS_META_PATH}")

    with EMBEDDINGS_META_PATH.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    metric = str(meta.get("metric", "ip")).lower()
    if metric not in {"ip", "inner_product", "l2", "euclidean"}:
        logger.warning("Unknown metric '%s', defaulting to inner-product.", metric)
        metric = "ip"

    _METRIC = "ip" if metric in {"ip", "inner_product"} else "l2"

    if "normalize" in meta:
        _NORMALIZE = bool(meta["normalize"])
    else:
        _NORMALIZE = _METRIC == "ip"

    _META = meta
    return meta


# ================================================================
# Embedding Table + FAISS
# ================================================================
def load_embedding_table() -> pd.DataFrame:
    global _EMB_TABLE

    if _EMB_TABLE is not None:
        return _EMB_TABLE

    if not EMBEDDINGS_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Missing: {EMBEDDINGS_PARQUET_PATH}")

    df = pd.read_parquet(EMBEDDINGS_PARQUET_PATH)

    required = ["post_id", "title", "subreddit", "score", "comments"]
    for col in required:
        if col not in df.columns:
            df[col] = "" if col not in {"score", "comments"} else 0

    df = df.sort_values("post_id").reset_index(drop=True)
    _EMB_TABLE = df
    return df


def load_faiss_index() -> faiss.Index:
    global _FAISS_INDEX

    if _FAISS_INDEX is not None:
        return _FAISS_INDEX

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing index: {FAISS_INDEX_PATH}")

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    df = load_embedding_table()
    if index.ntotal != len(df):
        logger.warning(
            "FAISS vectors (%d) != embedding rows (%d)",
            index.ntotal,
            len(df)
        )

    _FAISS_INDEX = index
    return index


# ================================================================
# Model Loader
# ================================================================
def load_model() -> SentenceTransformer:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    meta = _load_metadata()
    model_name = meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

    _MODEL = SentenceTransformer(model_name, device="cpu")
    return _MODEL


# ================================================================
# Query Encoding
# ================================================================
def encode_query(text: str) -> np.ndarray:
    if not text or not isinstance(text, str):
        raise ValueError("Query must be a non-empty string.")

    model = load_model()
    _load_metadata()

    emb = model.encode(
        [text],
        show_progress_bar=False,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )

    emb = np.asarray(emb, dtype="float32")
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    return emb


# ================================================================
# Convert FAISS distances → similarity scores
# ================================================================
def _convert_distances(distances: np.ndarray) -> np.ndarray:
    _load_metadata()
    if _METRIC == "ip":
        return distances  # already similarity scores
    return -distances     # invert L2


# ================================================================
# Semantic Search
# ================================================================
def semantic_search(
    query: str,
    k: int = 10,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if k <= 0:
        raise ValueError("k must be > 0")

    emb = encode_query(query)
    index = load_faiss_index()
    table = load_embedding_table()

    k = min(k, index.ntotal)
    distances, indices = index.search(emb, k)

    distances = distances[0]
    indices = indices[0]
    scores = _convert_distances(distances)

    results = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(table):
            continue

        if min_score is not None and score < min_score:
            continue

        row = table.iloc[idx]
        results.append({
            "post_id": int(row.post_id),
            "title": str(row.title),
            "subreddit": str(row.subreddit),
            "score": int(row.score),
            "comments": int(row.comments),
            "similarity_score": float(score),
            "url": row.get("url", "")
        })

    return results


# ================================================================
# FastAPI Router
# ================================================================
try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field

    class SearchRequest(BaseModel):
        query: str
        k: int = Field(10, ge=1, le=100)
        min_score: Optional[float] = None

    class SearchResponse(BaseModel):
        results: List[Dict[str, Any]]

    router = APIRouter(prefix="/api/search", tags=["semantic-search"])

    @router.post("/", response_model=SearchResponse)
    async def search_endpoint(payload: SearchRequest):
        try:
            results = semantic_search(
                query=payload.query,
                k=payload.k,
                min_score=payload.min_score,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {"results": results}

except ImportError:
    router = None


# ================================================================
# CLI for manual testing
# ================================================================
def _cli_demo():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser.add_argument("query", type=str)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    print("\n=== Semantic Search ===")
    print("Query:", args.query)
    results = semantic_search(args.query, k=args.k)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['similarity_score']:.4f}")
        print(f"  subreddit: {r['subreddit']}")
        print(f"  title    : {r['title']}")
        print(f"  url      : {r['url']}")
        print(f"  score    : {r['score']} | comments: {r['comments']}")


if __name__ == "__main__":
    _cli_demo()


__all__ = [
    "load_model",
    "load_embedding_table",
    "load_faiss_index",
    "encode_query",
    "semantic_search",
    "router",
]
