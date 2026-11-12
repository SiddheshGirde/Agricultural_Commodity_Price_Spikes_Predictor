"""
Phase 8 — feature engineering on price_sentiment_merged.csv.

Adds price-derived time-series features and resolves the sentiment-NaN
question flagged at the end of Phase 7. Does NOT generate the spike
target - that's Phase 9, kept separate per the original phase plan.

LEAKAGE RULE (same boundary as the diagram from message one): every
feature for day t uses only rows dated <= t, within that commodity's own
series. Nothing here ever reads price(t+1).

See accompanying notes for why rolling/lag features are trading-day
(row-based) rather than calendar-day, and why sentiment NaNs are imputed
to 0 rather than left missing - both are real modeling decisions with
named trade-offs, not neutral defaults.

Output: data/interim/price_sentiment_features.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from config import PROJECT_ROOT

INPUT_PATH = (PROJECT_ROOT / "data/interim/price_sentiment_merged.csv")
OUTPUT_PATH = (PROJECT_ROOT / "data/interim/price_sentiment_features.csv")

ROLLING_WINDOWS = [3, 7]  # trading days (rows), not calendar days


def engineer_price_features(g: pd.DataFrame) -> pd.DataFrame:
    """g = one commodity's rows only, pre-sorted by date. Every operation
    below is .shift()/.rolling()/.pct_change() with no negative offset
    anywhere, so by construction nothing here can read a future row."""
    g = g.copy()

    g["lag_1_price"] = g["price"].shift(1)
    g["pct_change_1d"] = g["price"].pct_change(1) * 100
    g["pct_change_3d"] = g["price"].pct_change(3) * 100

    for w in ROLLING_WINDOWS:
        g[f"rolling_mean_{w}d"] = g["price"].rolling(window=w).mean()
        g[f"rolling_std_{w}d"] = g["price"].rolling(window=w).std()

    # Momentum: today's price vs. its own 7-trading-day average.
    # Positive = trading above recent average, negative = below.
    g["momentum_7d"] = g["price"] - g["rolling_mean_7d"]

    # Free addition - min_price/max_price already exist in the input from
    # Phase 3, unused until now. Same-day spread, not a future value.
    g["intraday_price_range"] = g["max_price"] - g["min_price"]

    return g


def handle_sentiment_missingness(df: pd.DataFrame) -> pd.DataFrame:
    sentiment_cols = ["sentiment_positive_mean", "sentiment_neutral_mean",
                       "sentiment_negative_mean", "sentiment_dispersion"]
    df[sentiment_cols] = df[sentiment_cols].fillna(0.0)
    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values(["commodity", "date"]).reset_index(drop=True)

    frames = []
    for commodity in df["commodity"].unique():
        g = df[df["commodity"] == commodity].sort_values("date").reset_index(drop=True)
        frames.append(engineer_price_features(g))
    engineered = pd.concat(frames, ignore_index=True)

    engineered = handle_sentiment_missingness(engineered)

    engineered.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    print("\n=== Shape ===")
    print(engineered.shape)
    print("\n=== Columns ===")
    print(engineered.columns.tolist())

    # Expected NaNs from rolling warm-up only (first few rows per commodity)
    # - reported, not silently dropped. Same pattern applies to the other
    # rolling columns; checking one is representative.
    print("\n=== NaN count in rolling_mean_7d per commodity (expect ~6 per commodity, warm-up only) ===")
    print(engineered.groupby("commodity")["rolling_mean_7d"].apply(lambda s: s.isnull().sum()))

    # Independent leakage spot-check: recompute one rolling value by hand
    # and confirm it matches the pipeline's output exactly.
    sample_commodity = engineered["commodity"].iloc[0]
    sub = engineered[engineered["commodity"] == sample_commodity].reset_index(drop=True)
    check_row = 20
    manual_mean = sub["price"].iloc[check_row - 6:check_row + 1].mean()
    pipeline_mean = sub["rolling_mean_7d"].iloc[check_row]
    print(f"\nLeakage spot-check ({sample_commodity}, row {check_row}): "
          f"manual={manual_mean:.4f}, pipeline={pipeline_mean:.4f}, "
          f"match={np.isclose(manual_mean, pipeline_mean)}")

    print("\n=== Sentiment columns: confirm zero NaN remain after imputation ===")
    sentiment_cols = ["sentiment_positive_mean", "sentiment_neutral_mean",
                       "sentiment_negative_mean", "sentiment_dispersion"]
    print(engineered[sentiment_cols].isnull().sum())

    print(f"\n=== First 10 rows, {sample_commodity} ===")
    print(sub.head(10))