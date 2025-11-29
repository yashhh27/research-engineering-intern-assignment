# SimPPL Social Media Explorer: Investigative Dashboard
It fulfills all requirements of the SimPPL Research Engineer Intern assignment by integrating Semantic Search (OpenAI Embeddings), GenAI Summaries, and advanced Multimodal and Network analysis into an intuitive investigative tool.

Video_Demo: (https://drive.google.com/file/d/1v_7l2uXzMzHpa66vR1zNKbQhO8ys40d6/view?usp=sharing)



# System Design and Architecture (The "How")

The project uses a Microservice Simulation approach to solve the critical memory (OOM) constraint of cloud platforms:

1. UI Layer (streamlit_app/app.py): Handles filtering, layout, and user input. It remains lightweight.

2. Model/Data Layer (Serverless): All memory-intensive operations (embedding generation, model loading) are pushed to the OpenAI API via the src/data_loader.py module. This eliminates the fatal 1GB RAM crash by ensuring the server only stores the small, pre-calculated data arrays.

3. Modular Structure: The code is cleanly decoupled into specific directories (src/analytics, src/ui_components, src/llm_engine) for maintainability and future scaling.


# Local Setup & Run Instructions

Follow these steps to clone the repository and run the application locally.

Prerequisites

Python 3.9+

Git and Git LFS must be installed (due to the 103MB embedding file).
Step 1: Clone and Install Dependencies
1. Clone the repository (the Git LFS client handles the large embedding file)
git clone [https://github.com/yourusername/research-engineering-intern-assignment.git](https://github.com/yourusername/research-engineering-intern-assignment.git)
cd research-engineering-intern-assignment

2. Create and activate virtual environment
python -m venv venv
./venv/Scripts/activate  # On Windows PowerShell
source venv/bin/activate # On Linux/Mac

3. Install required libraries
pip install -r requirements.txt


Step 2: Configure OpenAI API Key

You must set your OpenAI API key so the app can generate new embeddings (if needed) and run the GPT-4 features.

Create a file named .env in your project root.

Add your secret key:

OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

 Step 3: Generate Initial Embeddings Cache (MANDATORY)

You must run the embedding generation locally once. This will use the OpenAI API to create the small, token-safe NumPy file (embeddings_openai.npy) that the live dashboard relies on.

 Run the app. The first run will automatically compute the embeddings
 (This process takes a few minutes and uses API credits)
streamlit run streamlit_app/app.py

 Step 4: Run the Application

The application will now load instantly on subsequent runs, relying on the newly created cache file.

streamlit run streamlit_app/app.py


# 📊 Features and Data Storytelling

The dashboard is structured into six tabs to tell a comprehensive story about the data.
 Tab 1: 📈 Trends & Volume

Feature: Posts Over Time (Tracks volume across subreddits) and Keyword Trends (Tracks frequency of key terms).

Story: Identifies the Temporal Trigger of a narrative and shows which communities were the earliest and largest amplifiers.


 Tab 2: 🧠 AI Briefing

Feature: AI Summary Generator (GPT-4 narrative report) and Offline Events Button (Google News/Wikipedia links).

Story: Synthesizes complex statistics into an executive summary and connects the online data to real-world events.


Tab 3: 🧬 Clusters & Topics

Feature: UMAP Projection and Sentiment by Cluster (Stacked bar chart).

Story: Uses AI (UMAP/K-Means) to identify how the main search topic fractures into sub-narratives, and provides the sentiment breakdown (Positive/Negative) of each discovered group.


 Tab 4: ⚔️ Controversy & Sources

Feature: Controversy Matrix (Score vs. Comments) and Source of Truth (Sunburst).

Story: Differentiates between Viral Hits (high engagement) and Controversial/Flame Wars (low approval, high debate). The Sunburst shows whether the narrative is driven by Original Discussion or External Links (Source Analysis).


Tab 5: 🕸️ Network

Feature: Author-Subreddit Network Graph.

Story: Visualizes the bipartite graph showing which Authors post into which Subreddits, revealing potential coordinated behavior or "super-spreaders" .


 Tab 6: 📸 Media Impact

Feature: Bar Chart of Media Distribution (Images, Links, Video, Text) and Image Gallery (for high-voted images).

Story: Fulfills the Multimodal Analysis requirement by answering which type of media (e.g., visual content vs. text) is most effective at generating high scores/engagement.