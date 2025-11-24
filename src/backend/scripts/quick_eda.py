from pathlib import Path
from src.backend.modules.data_loader import load_as_dataframe

def main():
    df = load_as_dataframe("data/data.jsonl")

    print("Rows, Cols:", df.shape)
    print("\nTime range:", df["timestamp"].min(), "→", df["timestamp"].max())

    print("\nTop 10 subreddits:")
    print(df["subreddit"].value_counts().head(10))

    daily = df.groupby(df["timestamp"].dt.date).size().reset_index(name="posts")
    print("\nDaily post volume (head):")
    print(daily.head())

    top_engaged = (
        df[["title", "subreddit", "score", "comments"]]
        .sort_values("score", ascending=False)
        .head(10)
    )
    print("\nTop 10 posts by score:")
    print(top_engaged.to_string(index=False))

if __name__ == "__main__":
    main()
