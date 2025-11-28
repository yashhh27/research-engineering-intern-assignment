import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import streamlit as st
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data.jsonl"
# 1. Define the persistent cache file
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "embeddings.npy" 

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



# =========================================================
# DATA LOADING + PREPROCESSING
# =========================================================

@st.cache_data(show_spinner=True)
def load_dataset() -> pd.DataFrame:
    """
    Loads and cleans the dataset from the JSONL file.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_json(DATA_PATH, lines=True)

    # Flatten 'data' wrapper if present
    if "data" in df.columns:
        df = pd.json_normalize(df["data"])

    # Detect timestamp
    ts_col = None
    for candidate in ["timestamp", "created_utc", "created_at"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        raise ValueError("No timestamp column found.")

    # Convert timestamp
    if ts_col == "created_utc":
        df[ts_col] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    df = df.dropna(subset=[ts_col]).copy()
    df.rename(columns={ts_col: "timestamp"}, inplace=True)

    # Detect text column
    if "selftext" in df.columns and df["selftext"].notna().any():
        text_col = "selftext"
    elif "title" in df.columns:
        text_col = "title"
    else:
        raise ValueError("No usable text column found.")

    df["text_col"] = df[text_col].fillna("").astype(str)

    # Time derived columns
    df["date"] = df["timestamp"].dt.date
    df["year_week"] = df["timestamp"].dt.strftime("%Y-%U")
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour

    # Optional fields defaults
    if "subreddit" not in df.columns: df["subreddit"] = "unknown"
    if "score" not in df.columns: df["score"] = 0
    if "comments" not in df.columns and "num_comments" in df.columns:
        df["comments"] = df["num_comments"]
    elif "comments" not in df.columns:
        df["comments"] = 0
    if "title" not in df.columns:
        df["title"] = df["text_col"].str.slice(0, 80)

    return df

# --- EMBEDDINGS MODEL ---
@st.cache_resource(show_spinner=False, ttl=3600, max_entries=1)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def get_or_create_embeddings(texts: List[str]) -> np.ndarray:
    """
    FOR CLOUD DEPLOYMENT: Forces the app to load embeddings from disk (npy).
    Will crash gracefully with an error if the file is missing or corrupted.
    """
    
    # 1. Check if the committed file is present
    if not EMBEDDINGS_FILE.exists():
        st.error(f"⚠️ Embeddings file not found on server: {EMBEDDINGS_FILE}")
        st.info("CRITICAL: You must commit 'embeddings.npy' to your data folder.")
        # CRASH GRACEFULLY: Stop here to prevent memory spike
        st.stop() 

    # 2. Load the file from disk
    try:
        embs = np.load(EMBEDDINGS_FILE)
    except Exception as e:
        st.error(f"❌ Error loading cached file. File may be corrupted. Error: {e}")
        st.stop()
    
    # 3. Check Data Integrity
    if len(embs) != len(texts):
        st.error(f"❌ Embeddings size mismatch! Cached file has {len(embs)} rows, but data has {len(texts)}.")
        st.info("Please regenerate 'embeddings.npy' locally and push the new file.")
        st.stop()

    # If all checks pass
    return embs
