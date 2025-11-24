from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.backend.modules.data_loader import load_as_dataframe


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]  # repo root

DATA_PATH = PROJECT_ROOT / "data" / "data.jsonl"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

METADATA_JSON_PATH = ARTIFACTS_DIR / "metadata.json"
METADATA_MD_PATH = ARTIFACTS_DIR / "metadata.md"
DAILY_COUNTS_CSV_PATH = ARTIFACTS_DIR / "daily_counts.csv"


# ---------------------------------------------------------------------------
# Core Stats
# ---------------------------------------------------------------------------

def _compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    total_posts = int(len(df))

    if df.empty or "timestamp" not in df.columns:
        return {"total_posts": total_posts, "start_date": None, "end_date": None}

    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()

    return {
        "total_posts": total_posts,
        "start_date": start_ts.date().isoformat() if pd.notnull(start_ts) else None,
        "end_date": end_ts.date().isoformat() if pd.notnull(end_ts) else None,
    }


def _compute_top_subreddits(df: pd.DataFrame, limit: int = 10) -> Dict[str, int]:
    if df.empty or "subreddit" not in df.columns:
        return {}

    return {
        name: int(count)
        for name, count in df["subreddit"].value_counts().head(limit).items()
    }


def _compute_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["date", "posts"])

    # Convert timestamps → dates first
    df["_date"] = df["timestamp"].dt.date

    daily = (
        df.groupby("_date")
        .size()
        .reset_index(name="posts")
        .rename(columns={"_date": "date"})
    )

    return daily.sort_values("date")[["date", "posts"]]


def _compute_top_posts(df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    required = {"title", "score", "comments", "url"}
    if df.empty or not required.issubset(df.columns):
        return []

    top = (
        df.sort_values("score", ascending=False)
        .loc[:, ["title", "score", "comments", "url"]]
        .head(limit)
    )

    result = []
    for _, row in top.iterrows():
        result.append(
            {
                "title": str(row["title"]).strip(),
                "score": int(row["score"]),
                "comments": int(row["comments"]),
                "url": str(row["url"]),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Metadata Aggregation
# ---------------------------------------------------------------------------

def build_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    basic = _compute_basic_stats(df)
    top_subs = _compute_top_subreddits(df)
    daily_df = _compute_daily_counts(df)
    top_posts = _compute_top_posts(df)

    daily_json = [
    {
        "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
        "posts": int(c),
    }
    for d, c in zip(daily_df["date"], daily_df["posts"])
]

    metadata = {
        "total_posts": basic["total_posts"],
        "start_date": basic["start_date"],
        "end_date": basic["end_date"],
        "top_subreddits": top_subs,
        "daily_counts": daily_json,
        "top_posts_by_score": top_posts,
    }

    return metadata, daily_df


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _ensure_artifacts_dir():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def write_metadata_json(metadata: Dict[str, Any]):
    with METADATA_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)


def write_daily_counts_csv(daily_df: pd.DataFrame):
    if daily_df.empty:
        pd.DataFrame(columns=["date", "posts"]).to_csv(DAILY_COUNTS_CSV_PATH, index=False)
    else:
        df = daily_df.copy()
        df["date"] = df["date"].astype(str)
        df.to_csv(DAILY_COUNTS_CSV_PATH, index=False)


def write_metadata_markdown(metadata: Dict[str, Any]):
    lines = []

    lines.append("# Dataset Metadata Summary\n")
    lines.append(f"- **Total posts**: {metadata.get('total_posts', 0)}")
    lines.append(f"- **Date range**: {metadata.get('start_date')} → {metadata.get('end_date')}\n")

    # Subreddits
    lines.append("## Top Subreddits\n")
    if metadata["top_subreddits"]:
        for k, v in metadata["top_subreddits"].items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("_No subreddit data available._")
    lines.append("")

    # Top posts
    lines.append("## Top Posts by Score\n")
    top_posts = metadata["top_posts_by_score"]
    if top_posts:
        lines.append("| Title | Score | Comments | URL |")
        lines.append("|-------|-------|----------|-----|")
        for p in top_posts:
            lines.append(f"| {p['title']} | {p['score']} | {p['comments']} | {p['url']} |")
    else:
        lines.append("_No posts available._")

    METADATA_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _ensure_artifacts_dir()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = load_as_dataframe(DATA_PATH)
    metadata, daily_df = build_metadata(df)

    write_metadata_json(metadata)
    write_daily_counts_csv(daily_df)
    write_metadata_markdown(metadata)

    print("\n=== Metadata Build Complete ===")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Total posts: {metadata['total_posts']}")
    print(f"Date range: {metadata['start_date']} → {metadata['end_date']}")
    print(f"Artifacts written to: {ARTIFACTS_DIR}\n")


if __name__ == "__main__":
    main()
