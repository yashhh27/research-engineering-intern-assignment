import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import streamlit as st
from openai import OpenAI
from tqdm import tqdm
import os
import tiktoken

# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data.jsonl"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "embeddings_openai.npy" 
EMBED_MODEL_NAME = "text-embedding-3-small"

# =========================================================
# DATA LOADING + PREPROCESSING
# =========================================================

@st.cache_data(show_spinner=True)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_json(DATA_PATH, lines=True)

    if "data" in df.columns:
        df = pd.json_normalize(df["data"])

    ts_col = None
    for candidate in ["timestamp", "created_utc", "created_at"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        raise ValueError("No timestamp column found.")

    if ts_col == "created_utc":
        df[ts_col] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    df = df.dropna(subset=[ts_col]).copy()
    df.rename(columns={ts_col: "timestamp"}, inplace=True)

    if "selftext" in df.columns and df["selftext"].notna().any():
        text_col = "selftext"
    elif "title" in df.columns:
        text_col = "title"
    else:
        raise ValueError("No usable text column found.")

    df["text_col"] = df[text_col].fillna("").astype(str)

    df["date"] = df["timestamp"].dt.date
    df["year_week"] = df["timestamp"].dt.strftime("%Y-%U")
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour

    if "subreddit" not in df.columns: df["subreddit"] = "unknown"
    if "score" not in df.columns: df["score"] = 0
    if "comments" not in df.columns and "num_comments" in df.columns:
        df["comments"] = df["num_comments"]
    elif "comments" not in df.columns:
        df["comments"] = 0
    if "title" not in df.columns:
        df["title"] = df["text_col"].str.slice(0, 80)

    return df

# =========================================================
# OPENAI CLIENT
# =========================================================

def get_openai_client_internal():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (FileNotFoundError, KeyError):
            pass

    if not api_key:
        return None

    return OpenAI(api_key=api_key, timeout=300.0)

# =========================================================
# EMBEDDINGS (TOKEN-SAFE)
# =========================================================

@st.cache_resource(show_spinner=False)
def get_or_create_embeddings(texts: List[str]) -> np.ndarray:
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError("embeddings_openai.npy not found. Please precompute embeddings.")

    embs = np.load(EMBEDDINGS_FILE)

    if len(embs) != len(texts):
        raise ValueError(
            f"Embedding count mismatch. Embeddings: {len(embs)}, Texts: {len(texts)}"
        )

    return embs

