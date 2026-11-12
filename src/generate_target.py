"""
Phase 9 — generate the spike target from price_sentiment_features.csv.

target = 1 if the NEXT REPORTED row's price is >=5% higher than today's,
else 0. "Next reported" = next trading day for that commodity's benchmark
market, not next calendar day (same convention used throughout - see the
Phase 8 note on the 07-05 -> 07-09 gap for why this matters).

This is the ONE deliberate exception to the "no future information" rule:
the label is allowed to look forward by definition. Features (X_t) never
do - that boundary was enforced entirely in features.py, upstream of this
script, using only .shift(1)/.rolling()/.pct_change(1) with positive
offsets. This file only ever reads shift(-1) to build y_t, never to build
a feature column.

The LAST row of each commodity's series has no "next" row - its target is
left as NaN and MUST be dropped before training (a NaN label can't be
learned from). Rows are not silently deleted here; they're flagged, and
the caller is responsible for the drop before Phase 10.

Output: data/interim/price_sentiment_with_target.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from config import PROJECT_ROOT

INPUT_PATH = (PROJECT_ROOT / "data/interim/price_sentiment_features.csv")
OUTPUT_PATH = (PROJECT_ROOT / "data/interim/price_sentiment_with_target.csv")

SPIKE_THRESHOLD_PCT = 5.0


def generate_target(g: pd.DataFrame) -> pd.DataFrame:
    """g = one commodity's rows only, pre-sorted by date."""
    g = g.copy()
    g["next_price"] = g["price"].shift(-1)
    g["next_date"] = g["date"].shift(-1)

    g["pct_change_to_next"] = (g["next_price"] - g["price"]) / g["price"] * 100
    g["days_to_next_report"] = (g["next_date"] - g["date"]).dt.days

    g["spike"] = np.where(
        g["next_price"].isna(), np.nan,
        (g["pct_change_to_next"] >= SPIKE_THRESHOLD_PCT).astype(float)
    )
    return g


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values(["commodity", "date"]).reset_index(drop=True)

    frames = [generate_target(df[df["commodity"] == c].sort_values("date").reset_index(drop=True))
              for c in df["commodity"].unique()]
    df = pd.concat(frames, ignore_index=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Rows with no target (last row per commodity - must be dropped before training) ===")
    print(df.groupby("commodity")["spike"].apply(lambda s: s.isna().sum()))

    # THE number this whole phase exists to produce - real class balance,
    # not the estimate from the architecture check.
    print("\n=== Class balance (spike=1) among rows WITH a valid target ===")
    valid = df.dropna(subset=["spike"])
    balance = valid.groupby("commodity")["spike"].agg(
        total="size",
        spikes="sum",
        spike_rate=lambda s: f"{s.mean()*100:.2f}%"
    )
    print(balance)

    # Quantify the trading-gap caveat flagged in Phase 8 - how many labeled
    # "1-day" spikes actually span more than 1 calendar day?
    print("\n=== Gap check: spike=1 rows where the 'next day' was actually >1 calendar day away ===")
    spikes_only = valid[valid["spike"] == 1]
    print(spikes_only.groupby("commodity")["days_to_next_report"].agg(
        multi_day_gap_spikes=lambda s: (s > 1).sum(),
        total_spikes="size",
        max_gap="max"
    ))

    # Independent spot-check: manually recompute one row's target, confirm
    # it matches the pipeline exactly - same discipline as the Phase 8
    # leakage check.
    sample_commodity = df["commodity"].iloc[0]
    sub = df[df["commodity"] == sample_commodity].reset_index(drop=True)
    check_row = 10
    manual_pct = (sub["price"].iloc[check_row + 1] - sub["price"].iloc[check_row]) / sub["price"].iloc[check_row] * 100
    manual_spike = 1.0 if manual_pct >= SPIKE_THRESHOLD_PCT else 0.0
    print(f"\nTarget spot-check ({sample_commodity}, row {check_row}): "
          f"manual_pct={manual_pct:.4f}, pipeline_pct={sub['pct_change_to_next'].iloc[check_row]:.4f}, "
          f"manual_spike={manual_spike}, pipeline_spike={sub['spike'].iloc[check_row]}, "
          f"match={np.isclose(manual_pct, sub['pct_change_to_next'].iloc[check_row])}")

    print(f"\n=== First 10 rows, {sample_commodity} (key columns only) ===")
    print(sub[["date", "price", "next_price", "next_date", "days_to_next_report", "pct_change_to_next", "spike"]].head(10))