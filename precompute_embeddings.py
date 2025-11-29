import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
import os
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
# ==========================
# CONFIG (same as app)
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT/"streamlit_app" / "data" / "data.jsonl"
EMBEDDINGS_FILE = PROJECT_ROOT / "streamlit_app" / "data" / "embeddings_openai.npy"
EMBED_MODEL_NAME = "text-embedding-3-small"

# ==========================
# OPENAI CLIENT
# ==========================

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_json(DATA_PATH, lines=True)

if "data" in df.columns:
    df = pd.json_normalize(df["data"])

if "selftext" in df.columns and df["selftext"].notna().any():
    texts = df["selftext"].fillna("").astype(str).tolist()
elif "title" in df.columns:
    texts = df["title"].fillna("").astype(str).tolist()
else:
    raise RuntimeError("No usable text column")

# ==========================
# TOKEN SAFE EMBEDDING
# ==========================

tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 7000

def token_length(text):
    return len(tokenizer.encode(text))

def token_chunker(texts):
    batch, tokens = [], 0

    for t in texts:
        if not t or not t.strip():
            t = "[NO_CONTENT]"

        t = t.replace("\n", " ").replace("\t", " ").strip()
        t_len = token_length(t)

        if t_len > MAX_TOKENS:
            t = tokenizer.decode(tokenizer.encode(t)[:MAX_TOKENS])
            t_len = MAX_TOKENS

        if tokens + t_len > MAX_TOKENS:
            yield batch
            batch, tokens = [], 0

        batch.append(t)
        tokens += t_len

    if batch:
        yield batch

# ==========================
# EMBEDDING LOOP
# ==========================

client = get_openai_client()
all_embeddings = []

for batch in tqdm(list(token_chunker(texts)), desc="Embedding batches"):
    response = client.embeddings.create(
        model=EMBED_MODEL_NAME,
        input=batch
    )
    all_embeddings.extend([e.embedding for e in response.data])

embeddings = np.array(all_embeddings)

assert embeddings.shape[0] == len(texts), "Embedding count mismatch"

np.save(EMBEDDINGS_FILE, embeddings)

print("✅ Embeddings saved:", EMBEDDINGS_FILE)
print("✅ Shape:", embeddings.shape)
