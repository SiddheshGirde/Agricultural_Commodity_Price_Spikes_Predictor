"""
Phase 7 — merge validated daily prices with aggregated daily sentiment.

Left-joins sentiment onto price (price is the spine — every trading day
must survive this merge, since Phase 9's target needs the full price
calendar). Days with no matching news get:
  - has_news = 0
  - article_count = 0
  - sentiment_positive_mean / neutral_mean / negative_mean / dispersion = NaN
    (left as missing, NOT filled with 0 — a computed 0 and "no data" are
    different things, and collapsing them would manufacture a fake
    "neutral" signal that never happened)
  - dominant_sentiment = "no_news" (an explicit fourth category, not
    silently folded into "neutral")

Does NOT engineer price-based features (rolling means, momentum, lag) or
generate the spike target — that's Phase 8/9. This is alignment only.

Output: data/interim/price_sentiment_merged.csv
"""

from pathlib import Path
import pandas as pd
from config import PROJECT_ROOT

PRICES_PATH = (PROJECT_ROOT / "data/interim/prices_clean.csv")
SENTIMENT_PATH = (PROJECT_ROOT / "data/interim/sentiment_daily.csv")
OUTPUT_PATH = (PROJECT_ROOT / "data/interim/price_sentiment_merged.csv")


def merge_price_and_sentiment(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    prices = prices.rename(columns={"Commodity": "commodity", "Arrival_Date": "date"})

    # Guard: prices_clean.csv should already be unique per (commodity, date)
    # from Phase 3's groupby — confirm rather than assume, since a silent
    # duplicate here would corrupt the merge (row explosion).
    price_dupes = prices.duplicated(subset=["commodity", "date"]).sum()
    if price_dupes > 0:
        raise ValueError(f"prices_clean.csv has {price_dupes} duplicate (commodity, date) rows — fix Phase 3 output first.")

    merged = prices.merge(sentiment, on=["commodity", "date"], how="left")

    if len(merged) != len(prices):
        raise ValueError(f"Merge changed row count: {len(prices)} price rows -> {len(merged)} merged rows. "
                          f"Sentiment data must have a duplicate (commodity, date) somewhere.")

    merged["has_news"] = merged["article_count"].notna().astype(int)
    merged["article_count"] = merged["article_count"].fillna(0).astype(int)
    merged["dominant_sentiment"] = merged["dominant_sentiment"].fillna("no_news")
    # sentiment_positive_mean / neutral_mean / negative_mean / dispersion:
    # deliberately NOT filled — stay NaN on no-news days.

    return merged.sort_values(["commodity", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, parse_dates=["Arrival_Date"])
    sentiment = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])

    merged = merge_price_and_sentiment(prices, sentiment)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    print("\n=== Shape ===")
    print(merged.shape)

    print("\n=== has_news rate by commodity ===")
    print(merged.groupby("commodity")["has_news"].agg(
        total_price_days="size",
        days_with_news="sum",
        news_rate=lambda s: f"{s.mean()*100:.1f}%"
    ))

    print("\n=== dominant_sentiment distribution (includes no_news) ===")
    print(merged.groupby("commodity")["dominant_sentiment"].value_counts())

    print("\n=== Missing-value counts (should exactly equal has_news==0 count, per commodity) ===")
    print(merged.groupby("commodity")[["sentiment_positive_mean", "sentiment_dispersion"]].apply(lambda g: g.isnull().sum()))

    print("\n=== Sample: one no-news day and one news day, per commodity ===")
    for commodity in merged["commodity"].unique():
        sub = merged[merged["commodity"] == commodity]
        print(f"\n-- {commodity} --")
        no_news_example = sub[sub["has_news"] == 0].head(1)
        news_example = sub[sub["has_news"] == 1].head(1)
        print(pd.concat([no_news_example, news_example])[
            ["date", "price", "has_news", "article_count", "sentiment_positive_mean", "dominant_sentiment"]
        ])