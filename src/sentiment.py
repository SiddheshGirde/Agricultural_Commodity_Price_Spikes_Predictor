"""
Phase 5 — FinBERT sentiment scoring for news_clean.csv.

Scores each headline for positive/neutral/negative probability, then applies
a targeted correction for a confirmed domain-gap bug: FinBERT (trained on
Financial PhraseBank, where "prices up" reads as bullish/good news) scores
headlines like "Onion prices shoot up by 52%" as positive, when for this
project that's exactly the negative/crisis signal being predicted. Found via
a 5-headline sanity check, confirmed at scale (307 "price up" headlines,
58.6% scored positive vs. 89.9% correct on "price down" headlines) — see
project notes for the full investigation. Scope of the fix: price-direction
language only. Output/export-volume language showing the same directional
ambiguity (e.g. "Soyabean exports up 50%") was identified but left
uncorrected — low volume (~2 headlines), and export vs. output growth push
commodity prices in opposite directions, so a blanket rule would introduce
a new, more subtle bug rather than fix this one.

Output: data/interim/news_with_sentiment.csv
Does NOT aggregate to daily features — that's Phase 6.
"""

import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import PROJECT_ROOT

NEWS_CLEAN_PATH = (PROJECT_ROOT / "data/interim/news_clean.csv")
OUTPUT_PATH = (PROJECT_ROOT / "data/interim/news_with_sentiment.csv")
MODEL_NAME = "ProsusAI/finbert"

# Requires "price"/"prices" within ~25 chars of rise-language, in either order,
# so it doesn't fire on unrelated uses of "up"/"rise" elsewhere in a headline.
# Non-capturing groups (?:...) — avoids the pandas capture-group UserWarning.
PRICE_RISE_PATTERN = re.compile(
    r"price[s]?\b.{0,25}\b(?:up|rise|rises|risen|shoot|shoots|surge|surges|soar|soars|jump|jumps|hike|hikes)\b"
    r"|"
    r"\b(?:up|rise|rises|risen|shoot|shoots|surge|surges|soar|soars|jump|jumps|hike|hikes)\b.{0,25}price[s]?\b",
    re.IGNORECASE,
)


def load_finbert():
    print("Loading FinBERT (first run downloads ~400MB, cached after that)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("Label mapping from model config:", model.config.id2label)

    device = 0 if torch.cuda.is_available() else -1
    print("Using GPU" if device == 0 else "Using CPU — fine at ~1,900 headlines, expect a few minutes")

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,       # return all 3 class scores, not just the top prediction
        truncation=True,
        max_length=512,
    )
    # If `top_k` errors on an older transformers version, use return_all_scores=True instead.


def score_headlines(sentiment_pipeline, texts, batch_size=16):
    results = sentiment_pipeline(texts, batch_size=batch_size)
    pos, neu, neg, label = [], [], [], []
    for r in results:
        scores = {item["label"].lower(): item["score"] for item in r}
        pos.append(scores.get("positive"))
        neu.append(scores.get("neutral"))
        neg.append(scores.get("negative"))
        label.append(max(r, key=lambda x: x["score"])["label"])
    return pos, neu, neg, label


def apply_price_rise_override(df: pd.DataFrame) -> pd.DataFrame:
    """Swap positive<->negative for headlines with price-rise language
    currently scored positive. Values come from the model's own confidence
    (not fabricated); flagged in a new column for transparency/reporting."""
    override_mask = (
        df["headline_text"].str.contains(PRICE_RISE_PATTERN, na=False, regex=True)
        & (df["sentiment_label"] == "positive")
    )
    print(f"Overriding {override_mask.sum()} headlines (price-rise language scored positive)")

    df.loc[override_mask, ["sentiment_positive", "sentiment_negative"]] = \
        df.loc[override_mask, ["sentiment_negative", "sentiment_positive"]].values
    df.loc[override_mask, "sentiment_label"] = "negative"
    df["price_direction_override_applied"] = override_mask
    return df


if __name__ == "__main__":
    sentiment_pipeline = load_finbert()

    df = pd.read_csv(NEWS_CLEAN_PATH)
    print(f"\n{len(df)} headlines loaded from {NEWS_CLEAN_PATH.name}")

    print("\n=== Sanity check: first 5 headlines ===")
    s_pos, s_neu, s_neg, _ = score_headlines(sentiment_pipeline, df["headline_text"].head(5).tolist())
    for text, p, n, ng in zip(df["headline_text"].head(5), s_pos, s_neu, s_neg):
        print(f"  '{text}' -> positive={p:.3f}, neutral={n:.3f}, negative={ng:.3f}")

    print(f"\nScoring all {len(df)} headlines...")
    pos, neu, neg, label = score_headlines(sentiment_pipeline, df["headline_text"].tolist())
    df["sentiment_positive"] = pos
    df["sentiment_neutral"] = neu
    df["sentiment_negative"] = neg
    df["sentiment_label"] = label

    missing = df[["sentiment_positive", "sentiment_neutral", "sentiment_negative"]].isnull().any(axis=1).sum()
    prob_sums = df["sentiment_positive"] + df["sentiment_neutral"] + df["sentiment_negative"]
    print(f"\nRows with an unmapped label (should be 0): {missing}")
    print(f"Probability sums — min: {prob_sums.min():.4f}, max: {prob_sums.max():.4f} (should both be ~1.0)")

    print("\n=== Sentiment distribution before override ===")
    print(df["sentiment_label"].value_counts())

    df = apply_price_rise_override(df)

    print("\n=== Sentiment distribution after override ===")
    print(df["sentiment_label"].value_counts())
    print("\n=== By commodity ===")
    print(df.groupby("commodity")["sentiment_label"].value_counts())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")