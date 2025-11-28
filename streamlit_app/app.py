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
    get_top_keywords
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
            df = load_dataset() # No args needed now, paths are handled in data_loader
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
    # 4. SEMANTIC SEARCH
    st.header("1. Semantic Search on Posts")
    
    # MOVED: Only initialize variable, don't load model yet
    all_embeddings = None 

    col_search, col_k = st.columns([3, 1])
    with col_search:
        query = st.text_input("Topic Search:", placeholder="e.g., housing crisis, tech layoffs")
    with col_k:
        top_k = st.number_input("Top K", 10, 500, 50)

    matched_df = pd.DataFrame()

    if query.strip():
        # LAZY LOAD: Model only loads here, AFTER user types
        with st.spinner("Loading AI model & Searching..."):
            if all_embeddings is None:
                all_embeddings = get_or_create_embeddings(df["text_col"].tolist())
                
            matched_df = run_semantic_search(df_filtered, all_embeddings, query, top_k=int(top_k))
            
            if not matched_df.empty:
                matched_df = attach_vader_sentiment(matched_df)

 

    if query.strip():
        with st.spinner(f"Searching for '{query}'..."):
            matched_df = run_semantic_search(df_filtered, all_embeddings, query, top_k=int(top_k))
            if not matched_df.empty:
                matched_df = attach_vader_sentiment(matched_df)

        if not matched_df.empty:
            st.success(f"Found {len(matched_df)} posts matching '{query}'")
            
            # Interactive Table
            st.subheader("Data Explorer")
            table_view = matched_df[['timestamp', 'subreddit', 'title', 'score', 'comments', 'sent_label', 'url']].copy()
            
            # Simple visual formatter
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
            
            # Keywords
            available_keywords = get_top_keywords(matched_df, top_n=50)
            if available_keywords:
                st.subheader("Keyword Trends")
                selected_keywords = st.multiselect("Select keywords:", options=available_keywords, default=available_keywords[:3])
                if selected_keywords:
                    # Quick keyword filtering logic
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
                sub_counts.columns = ["subreddit", "count"] # fix col names
                
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
            # 1. Header & Legend
            st.subheader("Controversy Matrix")
            st.markdown(
                """
                <div style="background-color: #1a1c24; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 20px; font-size: 14px;">
                <strong>💡 How to read this chart:</strong>
                <ul style="margin-top: 5px; margin-bottom: 0;">
                    <li>↗️ <strong>Top Right:</strong> Viral Hits.</li>
                    <li>↘️ <strong>Bottom Right:</strong> Controversial "Flame Wars".</li>
                    <li>👆 <strong>Click a dot</strong> to inspect details and get the link.</li>
                    <li>🔄 <strong>Double-click background</strong> to reset selection.</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 2. Data Preparation
            m_df = matched_df.copy()
            # CRITICAL: Reset index so Plotly click index matches Dataframe row index
            m_df = m_df.reset_index(drop=True) 
            
            if "upvote_ratio" not in m_df.columns: 
                m_df["upvote_ratio"] = 1.0
            
            # 3. Create Chart
            fig_cont = px.scatter(
                m_df, 
                x="comments", 
                y="score", 
                color="upvote_ratio", 
                log_x=True, 
                log_y=True, 
                color_continuous_scale="RdBu", 
                range_color=[0.5, 1.0], 
                title="Score vs Comments (Click points to inspect)", 
                hover_data=["title", "subreddit"]
            )
            
            # --- FIX: Custom Selection Styles ---
            # This ensures unselected dots don't disappear
            fig_cont.update_traces(
                # Highlight the selected dot (Big & Full Opacity)
                selected=dict(
                    marker=dict(
                        opacity=1.0, 
                        size=20,  # Made it slightly bigger to compensate for no border
                        color="red" # Optional: Change color to make it really pop
                    )
                ),
                # Keep unselected dots visible but dimmed
                unselected=dict(
                    marker=dict(
                        opacity=0.3
                    )
                )
            )
            
            # 4. Render Chart with Selection Enabled
            event = st.plotly_chart(
                fig_cont, 
                use_container_width=True,
                on_select="rerun",       # Reload app on click
                selection_mode="points"  # Single point selection
            )

            # 5. Handle Click Event
            if event and event.selection["points"]:
                # Get the index of the clicked dot
                point_index = event.selection["points"][0]["point_index"]
                
                # Retrieve data
                selected_row = m_df.iloc[point_index]
                
                # Display the "Open Post" card
                with st.container(border=True):
                    col_info, col_btn = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"**Selected:** {selected_row['title']}")
                        st.caption(f"r/{selected_row['subreddit']} • 💬 {selected_row['comments']} comments • ⬆️ {selected_row['score']}")
                    with col_btn:
                        st.link_button("🔗 Open Post", selected_row['url'], use_container_width=True)
            
            st.divider()
            
            st.subheader("Source of Truth")
            
            sb_df = matched_df.copy()
            if "domain" not in sb_df.columns: sb_df["domain"] = "self.reddit"
            sb_df["Content Category"] = sb_df["domain"].apply(categorize_domain)
            
            # 1. Add the Selectbox for Size Metric
            size_metric = st.selectbox("Segment Size By:", ["Count", "Comments", "Score"], key="sunburst_metric")
            
            # 2. Logic for Sizing
            if size_metric == "Count":
                sb_df["value_col"] = 1
            elif size_metric == "Score":
                sb_df["value_col"] = sb_df["score"].clip(lower=0) + 1
            else: # Comments
                sb_df["value_col"] = sb_df["comments"].clip(lower=0) + 1
            
            # 3. Create the Chart
            fig_sun = px.sunburst(
                sb_df, 
                path=["subreddit", "Content Category"], 
                values="value_col", 
                title=f"Content Composition by {size_metric}",
                color="Content Category" # Optional: keeps colors consistent
            )
            
            # 4. Increase Size and Readable Text
            fig_sun.update_layout(
                height=700, # Increased height
                uniformtext=dict(minsize=10, mode='hide')
            )
            fig_sun.update_traces(textinfo="label+percent entry")
            
            st.plotly_chart(fig_sun, use_container_width=True)

     
       # --- Tab 5: Network ---
        with tab_net:
            st.subheader("Network Analysis")
            
            with st.spinner("Mapping network..."):
                # Unpack all 4 return values
                fig_net, n_subs, n_auths, n_conns = build_author_subreddit_network(matched_df)
            
            # Create 3 columns explicitly
            col1, col2, col3 = st.columns(3)
            
            # Display metrics in each column
            col1.metric("Subreddits", n_subs)
            col2.metric("Authors", n_auths)
            col3.metric("Connections", n_conns)
            
            st.plotly_chart(fig_net, use_container_width=True)
        # --- Tab 6: Media Impact ---
        with tab_media:
            st.subheader("Multimodal Analysis: Media Impact")
            st.caption("Comparing how **Images**, **Videos**, and **Text** perform differently.")

            # 1. Apply Media Classification
            # We import the function here (or ensure it's imported at top)
            from src.analytics import detect_media_type
            
            media_df = matched_df.copy()
            media_df["media_type"] = media_df["url"].apply(detect_media_type)

            # 2. Insight: Which format goes viral?
            # Group by media type and get average score
            impact_stats = media_df.groupby("media_type")["score"].mean().reset_index()
            
            col_chart, col_gallery = st.columns([1, 2])
            
            with col_chart:
                st.markdown("##### 📊 Engagement by Format")
                fig_media = px.bar(
                    impact_stats, 
                    x="media_type", 
                    y="score", 
                    color="media_type",
                    title="Avg. Score per Media Type",
                    color_discrete_map={"Image": "#00CC96", "Video": "#AB63FA", "Text/Discussion": "#636EFA", "Link/Article": "#EF553B"}
                )
                st.plotly_chart(fig_media, use_container_width=True)
                
                # Mini Insight
                best_format = impact_stats.loc[impact_stats["score"].idxmax()]["media_type"]
                st.info(f"💡 **Insight:** **{best_format}** posts generate the highest engagement on average for this topic.")

            # 3. Media Gallery (The Visual "Wow" Factor)
            with col_gallery:
                st.markdown("##### 🖼️ Visual Gallery (Top Images)")
                
                # Filter for just images
                images = media_df[media_df["media_type"] == "Image"].sort_values("score", ascending=False).head(6)
                
                if not images.empty:
                    # Create a 3-column grid
                    cols = st.columns(3)
                    for idx, (index, row) in enumerate(images.iterrows()):
                        with cols[idx % 3]:
                            # Render the image
                            st.image(row["url"], use_container_width=True)
                            st.caption(f"**{row['score']}** ⬆️ | [View]({row['url']})")
                else:
                    st.warning("No image posts found in this search result.")

    # 6. FLOATING AI ANALYST
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Ready to analyze."}]
    
    # Generate suggestions if query changed
    if "last_query" not in st.session_state or st.session_state["last_query"] != query:
        st.session_state["ai_suggestions"] = generate_llm_suggestions(matched_df, query)
        st.session_state["last_query"] = query

    with st.popover("💬 AI Analyst"):
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
            
            # Suggestions inside container
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