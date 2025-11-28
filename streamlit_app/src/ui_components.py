import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

def render_dataset_overview(df):
    """
    Renders the 'Macro' view with Active Authors & Viral Metrics.
    OPTIMIZED: Uses lightweight groupby instead of resampling to prevent crashes.
    """
    st.header("📊 Dataset Overview")
    
    # --- 1. Calculate Statistics ---
    total_posts = len(df)
    
    if not df.empty:
        active_subs = df['subreddit'].nunique()
        active_authors = df['author'].nunique()
        viral_posts = len(df[df['score'] > 100])
    else:
        active_subs, active_authors, viral_posts = 0, 0, 0

    # --- 2. Custom CSS ---
    st.markdown("""
    <style>
        .metric-container {
            display: flex;
            justify-content: space-between;
            gap: 15px;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #1E1E1E;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            flex: 1;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            border-color: #666;
        }
        .metric-icon {
            font-size: 26px;
            margin-bottom: 10px;
            display: inline-block;
            padding: 10px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.05);
        }
        .metric-label {
            color: #AAAAAA;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .metric-value {
            color: #FFFFFF;
            font-size: 32px;
            font-weight: 800;
        }
        .card-blue { border-top: 4px solid #4DA6FF; }
        .card-purple { border-top: 4px solid #AB63FA; }
        .card-orange { border-top: 4px solid #FF9F1C; }
        .card-red { border-top: 4px solid #FF4B4B; }
    </style>
    """, unsafe_allow_html=True)

    # --- 3. Render HTML Cards ---
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card card-blue">
            <div class="metric-icon">📝</div>
            <div class="metric-label">Total Posts</div>
            <div class="metric-value">{total_posts:,}</div>
        </div>
        <div class="metric-card card-purple">
            <div class="metric-icon">👥</div>
            <div class="metric-label">Active Authors</div>
            <div class="metric-value">{active_authors:,}</div>
        </div>
        <div class="metric-card card-orange">
            <div class="metric-icon">🔥</div>
            <div class="metric-label">Viral Posts (>100)</div>
            <div class="metric-value">{viral_posts:,}</div>
        </div>
        <div class="metric-card card-red">
            <div class="metric-icon">💬</div>
            <div class="metric-label">Subreddits</div>
            <div class="metric-value">{active_subs}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 4. Charts (CRASH FIX HERE) ---
    if not df.empty:
        try:
            # FIX: Use groupby('date') instead of set_index().resample()
            # This avoids creating a massive memory-hogging index
            daily_counts = df.groupby("date").size().reset_index(name="posts")
            
            # Area Chart
            fig_global = px.area(
                daily_counts, x='date', y='posts',
                title="📈 Global Activity Volume",
                labels={'date': 'Date', 'posts': 'Volume'},
                template="plotly_dark",
                height=300
            )
            fig_global.update_traces(fillcolor="rgba(77, 166, 255, 0.2)", line=dict(color="#4DA6FF", width=2))
            
            # Peak Annotation
            if not daily_counts.empty:
                peak_row = daily_counts.loc[daily_counts['posts'].idxmax()]
                fig_global.add_annotation(
                    x=peak_row['date'], y=peak_row['posts'],
                    text=f"Peak: {peak_row['posts']}", showarrow=True, arrowhead=2,
                    arrowcolor="#FF4B4B", ax=0, ay=-40, bgcolor="#1E1E1E", bordercolor="#FF4B4B"
                )

            st.plotly_chart(fig_global, use_container_width=True)

            # Weekly & Subreddit Charts
            col_charts_1, col_charts_2 = st.columns(2)
            with col_charts_1:
                df['day_name'] = df['timestamp'].dt.day_name()
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_counts = df['day_name'].value_counts().reindex(day_order).reset_index()
                day_counts.columns = ['Day', 'Count']
                
                fig_days = px.bar(
                    day_counts, x='Day', y='Count', title="📅 Weekly Rhythm",
                    template="plotly_dark", color='Count', color_continuous_scale='Viridis', height=350
                )
                fig_days.update_layout(showlegend=False, xaxis_title=None)
                st.plotly_chart(fig_days, use_container_width=True)

            with col_charts_2:
                top_subs = df['subreddit'].value_counts().nlargest(10).reset_index()
                top_subs.columns = ['subreddit', 'count']
                top_subs = top_subs.sort_values('count', ascending=True)

                fig_subs = px.bar(
                    top_subs, x='count', y='subreddit', orientation='h', title="🏆 Top 10 Active Subreddits",
                    template="plotly_dark", color='count', color_continuous_scale='Blues', text='count', height=350
                )
                fig_subs.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_subs, use_container_width=True)
        
        except Exception as e:
            st.warning(f"Could not render charts: {e}")
            
    st.markdown("---")

def generate_context_links(query: str, peak_date_str: str) -> str:
    """
    Generates markdown links to Google News and Wikipedia.
    """
    if not peak_date_str or peak_date_str == "None":
        return ""

    try:
        p_date = pd.to_datetime(peak_date_str).date()
    except:
        return ""

    start_window = (p_date - timedelta(days=1)).strftime("%m/%d/%Y")
    end_window = (p_date + timedelta(days=1)).strftime("%m/%d/%Y")
    
    q_safe = query.replace(" ", "+")
    google_url = f"https://www.google.com/search?q={q_safe}&tbs=cdr:1,cd_min:{start_window},cd_max:{end_window}&tbm=nws"
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