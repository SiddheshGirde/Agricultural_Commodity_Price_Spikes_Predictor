"""
Phase 6 — aggregate headline-level sentiment into one row per (commodity, date).

Handles two edge cases named back in the original alignment discussion,
now implemented for real:
  - Dispersion (std dev) only means something with 2+ articles; single-
    article days get an explicit 0 rather than a std-of-0 that looks like
    "no disagreement" when there was only one opinion to begin with.
  - Zero-news days aren't dropped here — they don't exist in this output
    at all yet, since this table only has rows where news occurred. Phase 7
    is responsible for left-joining this onto the full price calendar and
    filling the resulting gaps with has_news=0, not this script.

Output: data/interim/sentiment_daily.csv
"""

from pathlib import Path
import pandas as pd
from config import PROJECT_ROOT

INPUT_PATH = (PROJECT_ROOT / "data/interim/news_with_sentiment.csv")
OUTPUT_PATH = (PROJECT_ROOT / "data/interim/sentiment_daily.csv")


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df["publish_date"] = pd.to_datetime(df["publish_date"])

    grouped = df.groupby(["commodity", "publish_date"])

    daily = grouped.agg(
        sentiment_positive_mean=("sentiment_positive", "mean"),
        sentiment_neutral_mean=("sentiment_neutral", "mean"),
        sentiment_negative_mean=("sentiment_negative", "mean"),
        article_count=("headline_text", "size"),
    ).reset_index()

    # Dominant label = mode. ties broken by earliest-appearing label in that
    # day's group, via value_counts' default ordering — acceptable for this
    # scale; not worth a fully custom tie-break for what will be a small
    # share of days.
    dominant = (
        grouped["sentiment_label"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index(name="dominant_sentiment")
    )
    daily = daily.merge(dominant, on=["commodity", "publish_date"])

    # Dispersion: std dev of positive-probability across that day's articles.
    # ddof=0 (population std) so a 1-article group returns exactly 0.0
    # rather than pandas' default ddof=1 producing NaN for n=1.
    dispersion = (
        grouped["sentiment_positive"]
        .agg(lambda s: s.std(ddof=0))
        .reset_index(name="sentiment_dispersion")
    )
    daily = daily.merge(dispersion, on=["commodity", "publish_date"])

    daily = daily.rename(columns={"publish_date": "date"})
    return daily.sort_values(["commodity", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    daily = aggregate_daily(df)

    daily.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    print("\n=== Shape ===")
    print(daily.shape)

    print("\n=== Per-commodity summary ===")
    print(daily.groupby("commodity").agg(
        news_days=("date", "size"),
        date_start=("date", "min"),
        date_end=("date", "max"),
        avg_articles_per_news_day=("article_count", "mean"),
        max_articles_in_one_day=("article_count", "max"),
        days_with_multiple_articles=("article_count", lambda s: (s > 1).sum()),
    ))

    print("\n=== Sanity check: dispersion should be exactly 0 on single-article days ===")
    single_article_days = daily[daily["article_count"] == 1]
    print(f"Single-article days: {len(single_article_days)}")
    print(f"Non-zero dispersion among them (should be 0): {(single_article_days['sentiment_dispersion'] != 0).sum()}")

    print("\n=== First 5 rows ===")
    print(daily.head())