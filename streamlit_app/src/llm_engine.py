import os
import pandas as pd  
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI() # Assumes api key in env
def generate_query_insight_style_summary(
    query: str,
    matched_df: pd.DataFrame,
    agg_posts_day: pd.DataFrame,
    sub_counts: pd.DataFrame,
) -> str:
    """Call GPT-4.1 to generate a narrative summary similar to the backend Query Insights."""

    total = len(matched_df)
    start = matched_df["timestamp"].min().date() if total > 0 else None
    end = matched_df["timestamp"].max().date() if total > 0 else None

    avg_score = float(matched_df["score"].mean()) if total > 0 else 0
    max_score = int(matched_df["score"].max()) if total > 0 else 0
    min_score = int(matched_df["score"].min()) if total > 0 else 0
    median_score = float(matched_df["score"].median()) if total > 0 else 0
    p90_score = float(np.percentile(matched_df["score"], 90)) if total > 0 else 0
    score_std = float(matched_df["score"].std()) if total > 0 else 0

    if not agg_posts_day.empty:
        peak_row = agg_posts_day.loc[agg_posts_day["posts"].idxmax()]
        peak_date = str(peak_row["plot_date"].date())
        peak_posts = int(peak_row["posts"])

        trough_row = agg_posts_day.loc[agg_posts_day["posts"].idxmin()]
        trough_date = str(trough_row["plot_date"].date())
        trough_posts = int(trough_row["posts"])
    else:
        peak_date = trough_date = None
        peak_posts = trough_posts = 0

    top_subs = sub_counts["subreddit"].head(3).tolist() if not sub_counts.empty else []

    prompt = f"""
You are Arbiter, a Senior Data Scientist and Cultural Analyst specializing in Reddit discourse.
Your task is to synthesize the provided metadata into a high-density strategic intelligence brief.

CONTEXT:
- Query: "{query}"
- Dataset: {total} posts from {start} to {end}

BEHAVIORAL METRICS:
- Engagement Volatility: Score Std Dev {score_std} vs Avg Score {avg_score}.
- Viral Ceiling: Max Score {max_score} vs 90th Percentile {p90_score}.
- Baseline Attention: Median Score {median_score}.

TEMPORAL DYNAMICS:
- Peak Activity: {peak_date} ({peak_posts} posts).
- Trough Activity: {trough_date} ({trough_posts} posts).

COMMUNITY ARCHITECTURE:
- Top Subreddits: {top_subs}

INSTRUCTIONS:
Write a professional, executive-level narrative (approx. 250 words) that goes beyond reporting numbers to explaining *behavior*. Structure your response as follows:

1. **The Signal Strength**: Open with an assessment of whether this topic is a "Viral Burst," a "Sustained Siege," or "Niche Noise" based on the volume ({total}) and the timeline ({start} to {end}).
2. **Community Ecosystem**: Analyze the top subreddits. Do these communities represent a unified front, or is the topic fractured across opposing ideological silos (e.g., political vs. hobbyist vs. news)?
3. **Engagement Physics**: Interpret the score distribution.
   - If (Max >> 90th %ile) and (Std Dev is high): Explain this as "Winner-Take-All" viral dynamics where only a few posts break through.
   - If (Median is close to Average): Explain this as "High Consistency" community consensus.
4. **Temporal Trigger**: Hypothesize *why* the peak date ({peak_date}) occurred based on the specific context of the query "{query}" (e.g., was there a real-world event?).
5. **Strategic Conclusion**: Conclude with a single sentence on the "Temperature" of this topic—is it heating up, cooling down, or highly polarized?

TONE:
- Analytic, sophisticated, and dense.
- Avoid filler phrases like "The data suggests." State observations directly.
- STRICT RULE: Do not hallucinate external events, but you may infer context based on the query name and the peak date provided.
"""

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are a senior data analyst specializing in digital narrative and social-media trend analysis.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return completion.choices[0].message.content


def generate_llm_suggestions(df, query):
    """
    Uses the LLM to generate 3 specific questions based on the search results.
    """
    if df.empty:
        return ["Summarize the main themes", "What is the sentiment trend?", "Identify key opinion leaders"]

    # Show the LLM a sneak peek of the data
    titles = df['title'].head(8).tolist()
    subreddits = df['subreddit'].unique().tolist()[:5]
    
    prompt = f"""
    Context: A user searched for "{query}" on Reddit. 
    Top Subreddits: {subreddits}
    Sample Post Titles: {titles}

    Task: Generate 3 short, analytical questions (max 10 words each) a data scientist would ask about this specific data.
    Return ONLY the 3 questions separated by a pipe character (|).
    Example: Why is sentiment negative?|Who are the top authors?|What happened on the peak date?
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        content = response.choices[0].message.content
        questions = [q.strip() for q in content.split('|')]
        return questions[:3]
    except Exception:
        return ["Analyze the sentiment", "Who are the top authors?", "Summarize the controversy"]

# =========================================================
# NEW: STREAMING CHATBOT (Fixes "Slowness")
# =========================================================
def process_floating_query_stream(user_query, matched_df, chat_container):
    # 1. Calculate Context Stats
    if not matched_df.empty:
        max_score_idx = matched_df['score'].idxmax()
        viral_post = matched_df.loc[max_score_idx]
        viral_info = f"Viral: '{viral_post['title']}' (Score: {viral_post['score']})"
        top_sub = matched_df['subreddit'].value_counts().idxmax()
        stats = f"{viral_info}. Top Subreddit: r/{top_sub}. Total Posts: {len(matched_df)}."
    else:
        stats = "No data available."

    data_snippet = matched_df[['title', 'subreddit', 'score', 'sent_label']].head(15).to_string()

    # 2. Build System Prompt
    system_prompt = (
        "You are Arbiter, an expert Reddit Data Analyst. "
        "Answer the user's question based STRICTLY on the provided data snippet and stats. "
        "Be concise, professional, and evidence-based. "
        f"Global Stats: {stats}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Data Snippet:\n{data_snippet}\n\nQuestion: {user_query}"}
    ]

    # 3. Update UI Immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with chat_container:
        st.chat_message("user").write(user_query)

    # 4. Stream Response
    with chat_container:
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                stream=True, # <--- ENABLE STREAMING
                temperature=0.3
            )
            response_text = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
