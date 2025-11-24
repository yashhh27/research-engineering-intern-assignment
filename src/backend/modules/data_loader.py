"""
System-level data ingestion module for the SimPPL social media analytics pipeline.

This module is responsible for robustly loading, normalizing, and cleaning
datasets originating from various JSON-based export formats. It provides:

Supported formats
-----------------
- `.jsonl` / `.ndjson`  : newline-delimited JSON, one object per line (streamed)
- `.json`               : either a list of objects or a single JSON object

Public API
----------
- load_records(path)      -> Iterator[dict]
    Lazily streams raw records from disk.

- load_all(path)          -> list[dict]
    Eagerly materializes all records into memory.

- load_as_dataframe(path) -> pandas.DataFrame
    Loads, flattens, normalizes, and cleans records into an ML-ready table
    with a stable schema and a `full_text` column for NLP work.

Design Goals
------------
- Never crash on malformed data (fault tolerance)
- Stable schema for ML + dashboards
- Memory-efficient (.jsonl streaming)
- Scalable for future Arrow / DuckDB / Polars use
- Predictable, explicit transformations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
Record = Dict[str, Any]


# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================

def _iter_jsonl(path: Path) -> Iterator[Record]:
    """
    Stream newline-delimited JSON (.jsonl/.ndjson) safely and efficiently.
    """
    with path.open("r", encoding="utf-8") as fh:
        for i, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d (%s): %s", i, path, exc)
                continue

            if not isinstance(obj, dict):
                logger.warning(
                    "Line %d in %s parsed JSON but not an object (got %r). Yielding anyway.",
                    i, path, type(obj),
                )
            yield obj


def _load_json(path: Path) -> Iterable[Record]:
    """
    Load regular JSON (.json). Accepts:
    - list of objects
    - a single object
    - any other value (wrapped for consistency)
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    logger.warning(
        "Top-level JSON in %s is %r; wrapping into a list.", path, type(data)
    )
    return [data]


# ===========================================================================
# PUBLIC LOADING API
# ===========================================================================

def load_records(path: PathLike) -> Iterator[Record]:
    """
    Lazily load raw JSON records from `.jsonl` / `.ndjson` or `.json`.

    This is the most memory-efficient entrypoint.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    suffix = p.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        yield from _iter_jsonl(p)
    elif suffix == ".json":
        for rec in _load_json(p):
            if isinstance(rec, dict):
                yield rec
            else:
                logger.warning("Skipping non-dict JSON value in %s: %r", p, type(rec))
    else:
        raise ValueError(
            f"Unsupported extension '{suffix}'. Supported: .jsonl, .ndjson, .json"
        )


def load_all(path: PathLike) -> List[Record]:
    """
    Eagerly load all records into memory.
    Suitable for moderate datasets that fit RAM.
    """
    return list(load_records(path))


# ===========================================================================
# NORMALIZATION + SCHEMA ALIGNMENT
# ===========================================================================

def _normalize_records(records: Iterable[Record]) -> pd.DataFrame:
    """Flatten nested JSON structures using pandas.json_normalize."""
    try:
        return pd.json_normalize(records)
    except Exception as exc:
        logger.error("json_normalize failed: %s", exc)
        return pd.DataFrame()


def _align_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert arbitrary flattened JSON into the canonical SimPPL schema.
    Missing columns are created (pd.NA).
    """
    schema_map = {
        "data.title":        "title",
        "data.selftext":     "body",
        "data.author":       "author",
        "data.subreddit":    "subreddit",
        "data.created_utc":  "timestamp",
        "data.url":          "url",
        "data.score":        "score",
        "data.num_comments": "comments",
    }

    aligned = pd.DataFrame()
    for raw_key, final_key in schema_map.items():
        if raw_key in df.columns:
            aligned[final_key] = df[raw_key]
        else:
            logger.debug("Missing %s; creating empty '%s'", raw_key, final_key)
            aligned[final_key] = pd.NA

    return aligned


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, type-coerce, and enrich the canonical schema.
    """
    # Timestamps → datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    # Text fields
    for col in ("title", "body", "author", "subreddit", "url"):
        df[col] = df[col].fillna("").astype(str)

    # Derived field for NLP
    df["full_text"] = (df["title"] + " " + df["body"]).str.strip()

    # Numeric fields
    for col in ("score", "comments"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


# ===========================================================================
# MASTER API
# ===========================================================================

def load_as_dataframe(path: PathLike) -> pd.DataFrame:
    """
    Load, normalize, and clean records into an ML-ready pandas DataFrame.
    """
    records = load_all(path)

    if not records:
        logger.warning("File %s contains no valid records; returning empty frame.", path)
        columns = [
            "title", "body", "author", "subreddit", "timestamp",
            "url", "score", "comments", "full_text",
        ]
        return pd.DataFrame(columns=columns)

    flattened = _normalize_records(records)
    aligned   = _align_schema(flattened)
    cleaned   = _clean_dataframe(aligned)

    return cleaned


# ===========================================================================
# DEFAULT DATASET LOCATION (NEW)
# ===========================================================================

def default_dataset_path() -> Path:
    """
    Return the canonical path to the dataset inside the project:

    project/
    ├── data/
    │   └── reddit.jsonl
    └── src/
        └── backend/modules/data_loader.py

    Returns
    -------
    Path
    """
    return Path(__file__).resolve().parents[2] / "data" / "data.jsonl"


__all__ = [
    "load_records",
    "load_all",
    "load_as_dataframe",
    "default_dataset_path",
]
