"""
Phase 3 - clean and standardize raw price and news data for every commodity in config.COMMODITIES.

Behaviour for the original three commodities is intended to be IDENTICAL to the earlier
hardcoded version (checked by running both on the same data). What is new:
  * commodities / benchmark markets / news patterns come from config.py
  * commodity and market names are whitespace-stripped (the live feed showed trailing spaces)
  * rows with a non-positive Modal_Price are dropped and COUNTED (a zero would corrupt the daily mean)
  * a configured commodity/market with no rows raises an error instead of silently producing nothing
  * headlines removed by a commodity's news_exclude rule are counted and shown
  * headlines tagged with more than one commodity are counted (deliberate: one row per match)
  * only yearly files inside the modeling window are read

Does NOT do sentiment scoring, merging, feature engineering or target generation.
Usage:  python src/preprocessing.py            (add --inspect to print sample headlines per commodity)
"""
import sys
from pathlib import Path

import pandas as pd

from config import (COMMODITIES, EXCLUDED_NEWS_CATEGORY, INTERIM_DIR, MODELING_END, MODELING_START,
                    NEWS_CSV_PATH, PRICE_PARQUET_DIR)

PRICE_COLUMNS = ["Commodity", "Market", "Variety", "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"]


def window_year_files(parquet_dir: Path):
    lo, hi = int(MODELING_START[:4]), int(MODELING_END[:4])
    files = [f for f in sorted(parquet_dir.glob("*.parquet")) if f.stem.isdigit() and lo <= int(f.stem) <= hi]
    if not files:
        raise FileNotFoundError(f"No yearly parquet files for {lo}-{hi} in {parquet_dir}")
    return files


def load_price_data(parquet_dir: Path) -> pd.DataFrame:
    """One row per (commodity, date) at the commodity's configured benchmark market: the mean
    across varieties reported that day."""
    wanted = {cfg["archive_name"] for cfg in COMMODITIES.values()}
    frames = []
    for f in window_year_files(parquet_dir):
        d = pd.read_parquet(f, columns=PRICE_COLUMNS)
        d["Commodity"] = d["Commodity"].str.strip()
        d["Market"] = d["Market"].str.strip()
        frames.append(d[d["Commodity"].isin(wanted)])
    df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # normalise the late-2025 "APMC" suffix (a no-op inside the 2013-2023 window; kept for robustness)
    df["base_market"] = df["Market"].str.replace(r"\s*APMC$", "", regex=True).str.strip()

    kept = []
    for name, cfg in COMMODITIES.items():
        sub = df[(df["Commodity"] == cfg["archive_name"]) & (df["base_market"] == cfg["archive_market"])]
        if sub.empty:
            near = sorted(df.loc[(df["Commodity"] == cfg["archive_name"]), "base_market"].value_counts().head(5).index)
            raise ValueError(f"No price rows for {name} at market '{cfg['archive_market']}'. "
                             f"Best-covered markets for it: {near}. Check config.py.")
        bad = (pd.to_numeric(sub["Modal_Price"], errors="coerce") <= 0) | sub["Modal_Price"].isna()
        if bad.any():
            print(f"{name}: dropped {int(bad.sum())} row(s) with a non-positive/missing Modal_Price")
        kept.append(sub[~bad])
    df = pd.concat(kept, ignore_index=True)
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

    return (df.groupby(["Commodity", "Arrival_Date"], as_index=False)
              .agg(price=("Modal_Price", "mean"), min_price=("Min_Price", "mean"),
                   max_price=("Max_Price", "mean"), n_varieties_reported=("Variety", "nunique")))


def load_and_tag_news(csv_path: Path, inspect: bool = False) -> pd.DataFrame:
    """Tag each headline with every configured commodity it mentions (one row per match - a headline
    naming two commodities appears twice, deliberately), drop international wire noise."""
    df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    df["publish_date"] = pd.to_datetime(df["publish_date"], format="%Y%m%d", errors="coerce")
    text = df["headline_text"]

    tagged = []
    for name, cfg in COMMODITIES.items():
        hit = text.str.contains(cfg["news_pattern"], case=False, na=False, regex=True)
        if cfg["news_exclude"]:
            drop = hit & text.str.contains(cfg["news_exclude"], case=False, na=False, regex=True)
            print(f"{name}: news_exclude rule removed {int(drop.sum())} of {int(hit.sum())} matching headlines")
            if inspect and drop.any():
                print("   e.g.", list(text[drop].sample(min(5, int(drop.sum())), random_state=7)))
            hit = hit & ~drop
        sub = df[hit].copy()
        sub["commodity"] = name
        tagged.append(sub)
    tagged = pd.concat(tagged, ignore_index=True)

    multi = int(tagged.duplicated(subset=["publish_date", "headline_text"], keep=False).sum())
    print(f"{multi} tagged row(s) come from headlines that mention more than one commodity (kept once per commodity)")

    before = len(tagged)
    tagged = tagged[tagged["headline_category"] != EXCLUDED_NEWS_CATEGORY].reset_index(drop=True)
    print(f"Excluded {before - len(tagged)} international-business wire headlines ({before} -> {len(tagged)})")
    return tagged


def apply_modeling_window(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    start, end = pd.Timestamp(MODELING_START), pd.Timestamp(MODELING_END)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].reset_index(drop=True)


if __name__ == "__main__":
    inspect = "--inspect" in sys.argv
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    prices = apply_modeling_window(load_price_data(PRICE_PARQUET_DIR), "Arrival_Date")
    prices.to_csv(INTERIM_DIR / "prices_clean.csv", index=False)
    print("\n=== Prices ===")
    print(prices.shape)
    print(prices.groupby("Commodity").agg(rows=("price", "size"), start=("Arrival_Date", "min"),
                                          end=("Arrival_Date", "max"),
                                          missing_price=("price", lambda s: s.isnull().sum())))

    news = apply_modeling_window(load_and_tag_news(NEWS_CSV_PATH, inspect), "publish_date")
    news.to_csv(INTERIM_DIR / "news_clean.csv", index=False)
    print("\n=== News ===")
    print(news.shape)
    print(news.groupby("commodity").agg(articles=("headline_text", "size"), unique_dates=("publish_date", "nunique"),
                                        start=("publish_date", "min"), end=("publish_date", "max")))
    if inspect:
        pd.set_option("display.width", 200, "display.max_colwidth", 110)
        for name in COMMODITIES:
            s = news[news["commodity"] == name]
            print(f"\n--- {name}: {len(s)} headlines, random sample ---")
            print(s.sample(min(12, len(s)), random_state=7)[["publish_date", "headline_text"]].to_string(index=False))