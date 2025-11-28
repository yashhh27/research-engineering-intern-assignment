import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
import pandas as pd
import plotly.express as px

# --- IMPORT MODULES ---
from src.data_loader import (
    load_dataset, 
    get_or_create_embeddings, 
    DATA_PATH, 
    EMBEDDINGS_FILE
)
from src.analytics import (
    run_semantic_search, 
    attach_vader_sentiment, 
    aggregate_time, 
    cluster_topics_and_umap, 
    sentiment_summary_by_cluster, 
    categorize_domain, 
    get_top_keywords,
    detect_media_type # Ensure this is imported
)
from src.viz import build_author_subreddit_network
from src.ui_components import render_dataset_overview, generate_context_links
from src.llm_engine import (
    generate_query_insight_style_summary, 
    generate_llm_suggestions, 
    process_floating_query_stream
)

# --- CONFIG ---
st.set_page_config(page_title="SimPPL – Social Media Explorer", layout="wide")

def main():
    st.title("SimPPL Social Media Explorer")
    st.caption("Semantic search + analytics directly on your cleaned Reddit dataset.")

    # 1. LOAD DATA
    with st.spinner("Loading dataset..."):
        try:
            df = load_dataset() 
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    # 2. GLOBAL FILTERS
    st.sidebar.header("Global Filters")
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()

    date_range = st.sidebar.date_input(
        "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )

    if len(date_range) != 2:
        st.info("⏳ Please select the end date to proceed.")
        st.stop()
    
    start_date, end_date = date_range
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    df_filtered = df[
        (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    ].copy()

    st.markdown("---")
    
    # 3. RENDER METRICS CARD
    render_dataset_overview(df_filtered)

    # 4. SEMANTIC SEARCH
    st.header("1. Semantic Search on Posts")
    
    all_embeddings = None 

    col_search, col_k = st.columns([3, 1])
    with col_search:
        query = st.text_input("Topic Search:", placeholder="e.g., housing crisis, tech layoffs", key="main_search")
    with col_k:
        top_k = st.number_input("Top K", 10, 500, 50)

    matched_df = pd.DataFrame()

    if query.strip():
        with st.spinner("Loading AI model & Searching..."):
            if all_embeddings is None:
                all_embeddings = get_or_create_embeddings(df["text_col"].tolist())
                
            matched_df = run_semantic_search(df_filtered, all_embeddings, query, top_k=int(top_k))
            
            if not matched_df.empty:
                matched_df = attach_vader_sentiment(matched_df)

        if not matched_df.empty:
            st.success(f"Found {len(matched_df)} posts matching '{query}'")
            
            # Interactive Table
            st.subheader("Data Explorer")
            table_view = matched_df[['timestamp', 'subreddit', 'title', 'score', 'comments', 'sent_label', 'url']].copy()
            
            table_view['sent_label'] = table_view['sent_label'].map({
                "positive": "🟢 Positive", "negative": "🔴 Negative", "neutral": "⚪ Neutral"
            })

            st.dataframe(
                table_view,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM YYYY, HH:mm", width="medium"),
                    "subreddit": st.column_config.TextColumn("Subreddit", width="small"),
                    "title": st.column_config.TextColumn("Title", width="large"),
                    "score": st.column_config.NumberColumn("Score", format="%d"),
                    "comments": st.column_config.NumberColumn("Comments", format="%d"),
                    "sent_label": st.column_config.TextColumn("Sentiment", width="small"),
                    "url": st.column_config.LinkColumn("Link", display_text="🔗", width="small")
                },
                hide_index=True, use_container_width=True, height=400
            )

    # 5. ANALYTICS TABS
    if not matched_df.empty:
        st.divider()
        st.markdown("### 📊 Analytics Dashboard")
        
        # DEFINITION OF TABS
        tab_trends, tab_ai, tab_topics, tab_deep, tab_net, tab_media = st.tabs([
            "📈 Trends & Volume", "🧠 AI Briefing", "🧬 Clusters & Topics", 
            "⚔️ Controversy & Sources", "🕸️ Network", "📸 Media Impact"
        ])

        # --- Tab 1: Trends ---
        with tab_trends:
            ts = matched_df.groupby(["date", "subreddit"]).size().reset_index(name="posts")
            ts["date"] = pd.to_datetime(ts["date"])
            fig_ts = px.line(ts, x="date", y="posts", color="subreddit", title="Posts Over Time", height=400)
            st.plotly_chart(fig_ts, use_container_width=True)
            
            available_keywords = get_top_keywords(matched_df, top_n=50)
            if available_keywords:
                st.subheader("Keyword Trends")
                selected_keywords = st.multiselect("Select keywords:", options=available_keywords, default=available_keywords[:3])
                if selected_keywords:
                    trend_data = []
                    temp_df = matched_df.copy()
                    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
                    for kw in selected_keywords:
                        mask = temp_df['title'].str.contains(kw, case=False, na=False)
                        res = temp_df[mask].set_index('timestamp').resample('W').size()
                        for d, c in res.items():
                            trend_data.append({"Date": d, "Keyword": kw, "Count": c})
                    if trend_data:
                        st.plotly_chart(px.line(pd.DataFrame(trend_data), x="Date", y="Count", color="Keyword", template="plotly_dark"), use_container_width=True)

        # --- Tab 2: AI Briefing ---
        with tab_ai:
            st.subheader("📝 Executive Briefing")
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("OPENAI_API_KEY not found.")
            elif st.button("Generate Narrative Report"):
                agg_day = aggregate_time(matched_df, "Day")
                sub_counts = matched_df["subreddit"].value_counts().reset_index(name="count")
                sub_counts.columns = ["subreddit", "count"]
                
                with st.spinner("Analyzing..."):
                    summary = generate_query_insight_style_summary(query, matched_df, agg_day, sub_counts)
                    st.markdown(summary)
                    if not agg_day.empty:
                        peak_date = str(agg_day.loc[agg_day["posts"].idxmax(), "plot_date"].date())
                        st.markdown(generate_context_links(query, peak_date), unsafe_allow_html=True)

        # --- Tab 3: Clusters ---
        with tab_topics:
            st.subheader("Topic Clusters")
            with st.spinner("Clustering..."):
                clusters, umap_df, centroids_df = cluster_topics_and_umap(matched_df)
            
            if clusters:
                if not umap_df.empty:
                    fig_umap = px.scatter(umap_df, x="x", y="y", color="cluster_label", hover_data=["title"], title="UMAP Projection")
                    st.plotly_chart(fig_umap, use_container_width=True)
                
                sent_df = sentiment_summary_by_cluster(umap_df)
                if not sent_df.empty:
                    st.plotly_chart(px.bar(sent_df, x="cluster", y=["Positive", "Negative", "Neutral"], title="Sentiment by Cluster"), use_container_width=True)

        # --- Tab 4: Deep Dive ---
        with tab_deep:
            st.subheader("Controversy Matrix")
            st.markdown("""
                <div style="background-color: #1a1c24; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 20px; font-size: 14px;">
                <strong>💡 How to read this chart:</strong>
                <ul style="margin-top: 5px; margin-bottom: 0;">
                    <li>↗️ <strong>Top Right:</strong> Viral Hits.</li>
                    <li>↘️ <strong>Bottom Right:</strong> Controversial "Flame Wars".</li>
                    <li>👆 <strong>Click a dot</strong> to see the link.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            m_df = matched_df.copy()
            m_df = m_df.reset_index(drop=True)
            if "upvote_ratio" not in m_df.columns: m_df["upvote_ratio"] = 1.0
            
            fig_cont = px.scatter(m_df, x="comments", y="score", color="upvote_ratio", log_x=True, log_y=True, color_continuous_scale="RdBu", range_color=[0.5, 1.0], title="Score vs Comments", hover_data=["title", "subreddit"])
            fig_cont.update_traces(
    # FIX: Removed the invalid 'line' dictionary entirely.
    selected=dict(
        marker=dict(
            opacity=1.0,
            size=15,
            # The border line property is not allowed here
        )
    ),
    # The unselected logic is correct
    unselected=dict(
        marker=dict(
            opacity=0.3
        )
    )
)
            event = st.plotly_chart(fig_cont, use_container_width=True, on_select="rerun", selection_mode="points")
            
            if event and event.selection["points"]:
                point_index = event.selection["points"][0]["point_index"]
                selected_row = m_df.iloc[point_index]
                with st.container(border=True):
                    col_info, col_btn = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"**Selected:** {selected_row['title']}")
                    with col_btn:
                        st.link_button("🔗 Open Post", selected_row['url'], use_container_width=True)

            st.divider()
            
            st.subheader("Source of Truth")
            sb_df = matched_df.copy()
            if "domain" not in sb_df.columns: sb_df["domain"] = "self.reddit"
            sb_df["Content Category"] = sb_df["domain"].apply(categorize_domain)
            
            col_opt, _ = st.columns([1, 3])
            with col_opt:
                size_metric = st.selectbox("Segment Size By:", ["Count", "Comments", "Score"], key="sb_metric")
            
            if size_metric == "Count": sb_df["value_col"] = 1
            elif size_metric == "Score": sb_df["value_col"] = sb_df["score"].clip(lower=0) + 1
            else: sb_df["value_col"] = sb_df["comments"].clip(lower=0) + 1
            
            fig_sun = px.sunburst(sb_df, path=["subreddit", "Content Category"], values="value_col", title=f"Content Composition by {size_metric}", color="Content Category")
            fig_sun.update_layout(height=750, uniformtext=dict(minsize=11, mode='hide'))
            st.plotly_chart(fig_sun, use_container_width=True)

        # --- Tab 5: Network ---
        with tab_net:
            st.subheader("Network Analysis")
            with st.spinner("Mapping..."):
                fig_net, n_s, n_a, n_c = build_author_subreddit_network(matched_df)
            col1, col2, col3 = st.columns(3)
            col1.metric("Subreddits", n_s)
            col2.metric("Authors", n_a)
            col3.metric("Connections", n_c)
            st.plotly_chart(fig_net, use_container_width=True)

        # --- Tab 6: Media Impact (Correctly Indented) ---
        with tab_media:
            st.subheader("Multimodal Analysis: Media Impact")
            media_df = matched_df.copy()
            media_df["media_type"] = media_df["url"].apply(detect_media_type)
            impact_stats = media_df.groupby("media_type")["score"].mean().reset_index()
            
            col_chart, col_gallery = st.columns([1, 2])
            with col_chart:
                fig_media = px.bar(impact_stats, x="media_type", y="score", color="media_type", title="Avg Engagement")
                st.plotly_chart(fig_media, use_container_width=True)
            
            with col_gallery:
                st.markdown("##### 📸 Image Gallery")
                images = media_df[media_df["media_type"] == "Image"].sort_values("score", ascending=False).head(6)
                if not images.empty:
                    cols = st.columns(3)
                    for idx, (index, row) in enumerate(images.iterrows()):
                        cols[idx % 3].image(row["url"], use_container_width=True)
                        cols[idx % 3].caption(f"Score: {row['score']}")
                else:
                    st.info("No images found in this result set.")

    # =========================================================
    # 6. FLOATING AI ANALYST
    # =========================================================
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Ready to analyze."}]
    
    if "last_query" not in st.session_state or st.session_state["last_query"] != query:
        st.session_state["ai_suggestions"] = generate_llm_suggestions(matched_df, query)
        st.session_state["last_query"] = query

    with st.popover("💬 AI Analyst"):
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
            
            if "ai_suggestions" in st.session_state and st.session_state["ai_suggestions"]:
                st.markdown("---")
                st.caption("💡 Suggested Questions:")
                for i, q in enumerate(st.session_state["ai_suggestions"]):
                    if st.button(q, key=f"sug_{i}", use_container_width=True):
                        process_floating_query_stream(q, matched_df, chat_container)
                        st.rerun()

        if prompt := st.chat_input("Ask Arbiter..."):
            process_floating_query_stream(prompt, matched_df, chat_container)

if __name__ == "__main__":
    main()