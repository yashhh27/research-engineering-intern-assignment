import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple
from sentence_transformers import util
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import timedelta
import re
from openai import OpenAI
# Import load_embed_model if needed for search, 
# or pass the model in. For now, you can import it:

# --- NEW HELPER: CLIENT GETTER (For internal use) ---
def get_openai_client_analytics():
    """Retrieves the OpenAI client (needed for query embedding)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def run_semantic_search(
    df: pd.DataFrame, 
    all_embeddings: np.ndarray, 
    query: str, 
    top_k: int = 50
) -> pd.DataFrame:
    """
    Performs semantic search safely, handling filtered dataframes.
    """
    client = get_openai_client_analytics()
    if not client:
        st.error("Semantic search failed: OpenAI client not initialized. Check API key in Secrets.")
        return pd.DataFrame()
    try:
        query_response = client.embeddings.create(
            # CRITICAL: Use the same model as defined in data_loader.py
            input=[query.replace("\n", " ").replace("\t", " ")],
            model="text-embedding-3-small" 
        )
        # Convert response object to a NumPy vector
        query_emb = np.array(query_response.data[0].embedding)
    except Exception as e:
        st.error(f"Error generating query embedding: {e}")
        return pd.DataFrame()

    # --- CRITICAL FIX: DATA ALIGNMENT ---
    # Case 1: User is searching the FULL dataset
    if len(df) == len(all_embeddings):
        active_embeddings = all_embeddings
        
    # Case 2: User is searching a FILTERED subset (e.g. by date)
    # We must slice the master embeddings array to match the current dataframe rows
    else:
        # We assume df.index corresponds to the original indices in all_embeddings
        # This requires that you DID NOT do df.reset_index(drop=True) during filtering
        try:
            active_embeddings = all_embeddings[df.index]
        except IndexError:
            # Fallback if indices are messed up
            st.error("Index mismatch error. Please clear filters and try again.")
            return pd.DataFrame()

    # Compute Dot Product (Faster than Cosine Sim if normalized)
    scores = np.dot(active_embeddings, query_emb)

    # Get Top K
    # We use min() to handle cases where the filtered result < Top K
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]
    

    # Retrieve rows using iloc on the ACTIVE subset
    # Since active_embeddings corresponds 1:1 with df, we can use iloc directly
    subset = df.iloc[top_indices].copy()
    subset["similarity_score"] = top_scores.astype(float)
    
    return subset.sort_values("similarity_score", ascending=False)




def attach_vader_sentiment(df_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Vader sentiment to each row as:
    - compound (float)
    - sent_label ('positive' | 'negative' | 'neutral')
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        # Fallback: neutral sentiment if library not installed
        df_out = df_slice.copy()
        df_out["compound"] = 0.0
        df_out["sent_label"] = "neutral"
        return df_out

    analyzer = SentimentIntensityAnalyzer()

    compounds: List[float] = []
    labels: List[str] = []

    for _, row in df_slice.iterrows():
        title = str(row.get("title") or "")
        body = str(row.get("selftext") or "")
        text = (title + " " + body).strip()
        if not text:
            c = 0.0
        else:
            c = float(analyzer.polarity_scores(text)["compound"])
        compounds.append(c)

        if c > 0.05:
            labels.append("positive")
        elif c < -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")

    df_out = df_slice.copy()
    df_out["compound"] = compounds
    df_out["sent_label"] = labels
    return df_out

# =========================================================
# TIME AGGREGATION (for time-series + GPT)
# =========================================================

def aggregate_time(df: pd.DataFrame, granularity: str = "Day") -> pd.DataFrame:
    """
    Aggregate posts over time.
    granularity ∈ {"Day", "Week", "Month"}.
    Returns DataFrame with ['plot_date', 'posts'].
    """
    if df.empty:
        return pd.DataFrame(columns=["plot_date", "posts"])

    if granularity == "Day":
        grouped = df.groupby("date").size().reset_index(name="posts")
        grouped["plot_date"] = pd.to_datetime(grouped["date"])
    elif granularity == "Week":
        grouped = df.groupby("year_week").size().reset_index(name="posts")
        grouped["plot_date"] = pd.to_datetime(grouped["year_week"] + "-0", format="%Y-%U-%w")
    else:  # "Month"
        grouped = df.groupby("year_month").size().reset_index(name="posts")
        grouped["plot_date"] = pd.to_datetime(grouped["year_month"] + "-01")

    return grouped.sort_values("plot_date")

# =========================================================
# TOPIC CLUSTERING (TF-IDF + KMeans) + UMAP
# =========================================================

def cluster_topics_and_umap(
    df_slice: pd.DataFrame,
    max_clusters: int = 5,
    max_terms: int = 6,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """
    Cluster titles of df_slice using TF-IDF + KMeans, then project with UMAP.
    Returns (clusters_list, umap_df, centroids_df).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
    except ImportError:
        return [], pd.DataFrame(), pd.DataFrame()

    if df_slice.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    # --- SAFETY: Ensure 'url' column exists ---
    if "url" not in df_slice.columns:
        df_slice["url"] = "#"

    # Use title as text basis for clustering
    titles = df_slice.get("title")
    if titles is None:
        return [], pd.DataFrame(), pd.DataFrame()

    # Keep non-empty titles
    titles_series = titles.fillna("").astype(str)
    non_empty_mask = titles_series.str.strip() != ""
    if non_empty_mask.sum() < 5:
        # too few posts for meaningful clustering
        return [], pd.DataFrame(), pd.DataFrame()

    df_used = df_slice[non_empty_mask].reset_index(drop=True)
    texts = titles_series[non_empty_mask].tolist()

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)

    # Heuristic: 1 cluster per ~8 posts, bounded by [2, max_clusters]
    n_clusters = max(2, min(max_clusters, (len(texts) // 8) or 2))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    feature_names = np.array(vectorizer.get_feature_names_out())

    # Attach Vader sentiment to df_used
    df_used = attach_vader_sentiment(df_used)

    clusters: List[Dict[str, Any]] = []

    for cid in range(n_clusters):
        idxs = np.where(labels == cid)[0]
        if idxs.size == 0:
            continue

        # Compute mean tf-idf vector per cluster and pick top terms
        centroid = X[idxs].mean(axis=0)
        centroid_vec = np.asarray(centroid).ravel()
        top_idx = centroid_vec.argsort()[::-1][:max_terms]
        top_terms = feature_names[top_idx].tolist()

        cluster_rows = df_used.iloc[idxs]
        size = int(len(cluster_rows))

        avg_c = float(cluster_rows["compound"].mean())
        pos_share = float((cluster_rows["sent_label"] == "positive").mean())
        neg_share = float((cluster_rows["sent_label"] == "negative").mean())
        neu_share = float((cluster_rows["sent_label"] == "neutral").mean())
        ex_titles = cluster_rows["title"].head(3).tolist()

        clusters.append(
            {
                "cluster_id": cid,
                "size": size,
                "top_terms": top_terms,
                "example_titles": ex_titles,
                "example_rows": cluster_rows.head(3),
                "avg_compound": avg_c,
                "positive_share": pos_share,
                "negative_share": neg_share,
                "neutral_share": neu_share,
            }
        )

    # UMAP projection
    try:
        import umap
    except ImportError:
        return clusters, pd.DataFrame(), pd.DataFrame()

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    emb_2d = reducer.fit_transform(X)

    umap_df = pd.DataFrame(
        {
            "x": emb_2d[:, 0],
            "y": emb_2d[:, 1],
            "cluster_id": labels.astype(int),
            # Use string labels for better categorical plotting
            "cluster_label": [f"Cluster {i}" for i in labels],
            "title": df_used["title"].values,
            "subreddit": df_used["subreddit"].values,
            "score": df_used["score"].values,
            "sent_label": df_used["sent_label"].values,
            "url": df_used["url"].values, # <--- Added URL here for the Galaxy Chart
        }
    )

    centroids_df = (
        umap_df
        .groupby("cluster_label", as_index=False)
        .agg(x=("x", "mean"), y=("y", "mean"))
    )

    return clusters, umap_df, centroids_df

def sentiment_summary_by_cluster(umap_df: pd.DataFrame) -> pd.DataFrame:
    """
     aggregations for the stacked bar chart.
    """
    if umap_df.empty:
        return pd.DataFrame()

    # Group by cluster and sentiment label
    counts = umap_df.groupby(["cluster_label", "sent_label"]).size().unstack(fill_value=0)
    
    # Ensure all sentiment columns exist (even if count is 0)
    for col in ["positive", "negative", "neutral"]:
        if col not in counts.columns:
            counts[col] = 0
            
    # Calculate percentages
    counts["total"] = counts.sum(axis=1)
    # Avoid division by zero
    counts = counts[counts["total"] > 0].copy()
    
    counts["Positive"] = counts["positive"] / counts["total"]
    counts["Negative"] = counts["negative"] / counts["total"]
    counts["Neutral"] = counts["neutral"] / counts["total"]
    
    # Format for Plotly
    summary = counts[["Positive", "Negative", "Neutral"]].reset_index()
    summary.rename(columns={"cluster_label": "cluster"}, inplace=True)
    
    return summary

def categorize_domain(domain: str) -> str:
    """
    Buckets raw domains into high-level content categories.
    """
    d = str(domain).lower().strip()
    
    # 1. Self/Original Content
    if d.startswith("self.") or "reddit.com" in d:
        return " Original Discussion"
    
    # 2. Native Reddit Media & Image Hosts
    if any(x in d for x in ["v.redd.it", "i.redd.it", "imgur", "giphy", "gfycat"]):
        return " Image/Media"
    
    # 3. Video Platforms
    if any(x in d for x in ["youtube", "youtu.be", "tiktok", "twitch", "vimeo"]):
        return " Video"
    
    # 4. Social Media
    if any(x in d for x in ["twitter", "x.com", "instagram", "facebook", "linkedin"]):
        return " Social Media"
    
    # 5. News & Articles (Catch-all for everything else)
    return " External Link / News"


def get_top_keywords(df, top_n=20):
    """
    Extracts high-quality keywords using TF-IDF and N-Grams.
    Removes noise (numbers, short words, common Reddit terms).
    """
    if df.empty:
        return []

    # 1. CLEANING FUNCTION
    # Remove numbers, special characters, and short words
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)  # Remove numbers (fixes "000")
        text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove 1-2 letter words
        text = re.sub(r'http\S+', '', text) # Remove URLs
        return text

    # Apply cleaning
    clean_titles = df['title'].apply(clean_text)

    # 2. SMART EXTRACTION (TF-IDF + N-Grams)
    # ngram_range=(1, 2) captures single words AND 2-word phrases (e.g., "border patrol")
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  
        max_features=1000,
        # Ignore words that appear in >90% of posts (too common)
        max_df=0.90,
        # Ignore words that appear in <3 posts (typos/irrelevant)
        min_df=3 
    )

    try:
        tfidf_matrix = tfidf.fit_transform(clean_titles)
        
        # 3. SUM SCORES to find highest weighted keywords
        # We sum the TF-IDF scores, not just counts
        sum_scores = tfidf_matrix.sum(axis=0)
        
        # Map scores to words
        words_freq = [
            (word, sum_scores[0, idx]) 
            for word, idx in tfidf.vocabulary_.items()
        ]
        
        # Sort by score desc
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        # Return top N words
        return [w[0] for w in words_freq[:top_n]]

    except ValueError:
        return []
    
    
def generate_context_links(query: str, peak_date_str: str) -> str:
    """
    Generates markdown links to Google News and Wikipedia 
    filtered specifically for the peak activity date.
    """
    if not peak_date_str or peak_date_str == "None":
        return ""

    try:
        # Parse the date string back to a datetime object
        p_date = pd.to_datetime(peak_date_str).date()
    except:
        return ""

    # Create a 3-day window (Day before, Day of, Day after)
    start_window = (p_date - timedelta(days=1)).strftime("%m/%d/%Y")
    end_window = (p_date + timedelta(days=1)).strftime("%m/%d/%Y")
    
    # URL Encoding the query
    q_safe = query.replace(" ", "+")
    
    # Google Search with Date Filter (tbs=cdr...)
    # This syntax tells Google: "Search for X between Date A and Date B"
    google_url = f"https://www.google.com/search?q={q_safe}&tbs=cdr:1,cd_min:{start_window},cd_max:{end_window}&tbm=nws"
    
    # Wikipedia Current Events for that Month
    wiki_month = p_date.strftime("%B_%Y")
    wiki_url = f"https://en.wikipedia.org/wiki/Portal:Current_events/{wiki_month}"

    return f"""
    <div style="background-color: #262730; padding: 15px; border-radius: 8px; border-left: 5px solid #FF4B4B; margin-top: 10px;">
        <h4 style="margin:0; padding-bottom:5px;">📅 Context Check: {peak_date_str}</h4>
        <p style="font-size: 14px; margin-bottom: 10px;">
            Investigate what caused this spike. These links search for <b>"{query}"</b> strictly around the peak date.
        </p>
        <a href="{google_url}" target="_blank" style="text-decoration: none;">
            <button style="background-color: #4285F4; color: white; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; margin-right: 10px;">
                🔍 Search Google News ({start_window} - {end_window})
            </button>
        </a>
        <a href="{wiki_url}" target="_blank" style="text-decoration: none;">
            <button style="background-color: #333; color: white; border: 1px solid #555; padding: 8px 12px; border-radius: 4px; cursor: pointer;">
                📖 Wikipedia Current Events
            </button>
        </a>
    </div>
    """
    
def detect_media_type(url: str) -> str:
    """
    Classifies a post as Image, Video, or Text based on the URL.
    """
    u = str(url).lower()
    if any(x in u for x in [".jpg", ".png", ".gif", "i.redd.it", "imgur.com"]):
        return "Image"
    elif any(x in u for x in ["youtube", "youtu.be", "v.redd.it", "twitch", "tiktok"]):
        return "Video"
    elif "reddit.com" in u and "comments" in u:
        return "Text/Discussion"
    else:
        return "Link/Article"
