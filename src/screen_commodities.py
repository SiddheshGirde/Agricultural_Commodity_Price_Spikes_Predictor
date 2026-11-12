"""
Commodity screening - READ-ONLY. Decides which additional commodities are worth
adding BEFORE any pipeline change, using the same lessons learned on the first three:

  * Wheat had 11 spike events in 10 years -> a held-out test set with 3 positives, no
    usable model. Spike COUNT (and test-set spikes) decides viability, not the commodity name.
  * Onion's "calm" days already moved ~6.7% on average, so a 5% spike barely stood out
    (F1 0.31). Soyabean's calm days moved ~2.8% (F1 0.66). Baseline noise is flagged.
  * News sentiment contributed little in SHAP, so news availability is a SOFT flag here,
    not a reason to exclude.

LESSON FROM THE FIRST RUN: spike rates depend heavily on WHICH market is used (onion: 6.25% at
the best-covered market, Bishnupur, vs 22.5% at Lasalgaon). So:
  * the three reference commodities are pinned to the markets we actually modeled
    (Lasalgaon / Indore / Kanpur (Grain)), so they reproduce the known numbers;
  * other commodities default to their best-COVERED market, which is only a first pass;
  * `--top-markets N` / `--include-markets` compare several markets for a shortlist so a
    benchmark can be chosen deliberately (large, well-covered, and present in the live feed).

--live-only fetches the live feed once per matched commodity and screens ONLY archive markets whose
name (formatting-insensitively, same key as the collector) appears in it. It is a single snapshot:
run it after 18:00, on several evenings - the day's feed fills in gradually, so a market missing
at 15:00 may appear at 20:00.

Verdicts: CANDIDATE (counts pass, calm days <= 5%), NOISY (counts pass but a 5% move is routine
noise - the onion problem), TOO-FEW-SPIKES (the wheat problem), TOO-SPARSE. Headline counts match
the commodity's base name (plurals and known spelling variants) and are a soft flag only.

Uses the same definitions as the pipeline: window 2013-07-01..2023-06-30, one price per
(commodity, date) = mean Modal_Price across varieties at that market, spike = next
REPORTED day's price >= +5%, chronological 80/20 split.

USAGE
  python src/screen_commodities.py                      # screen everything with enough coverage
  python src/screen_commodities.py --names tomato potato garlic
  python src/screen_commodities.py --names tomato potato --top-markets 8     # compare markets
  python src/screen_commodities.py --names tomato --include-markets azadpur  # plus a named market
  python src/screen_commodities.py --names tomato --live-only   # only markets present in TODAY's live feed
  python src/screen_commodities.py --no-news            # skip the headline check (faster)
Output: printed table + data/interim/commodity_screening.csv
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("PBL_ROOT") or Path(__file__).resolve().parents[1])
PARQUET_DIR = ROOT / "Data" / "parquet"
NEWS_PATH = ROOT / "data" / "raw" / "news" / "india-news-headlines.csv"
OUT_PATH = ROOT / "data" / "interim" / "commodity_screening.csv"

WINDOW_START, WINDOW_END = "2013-07-01", "2023-06-30"
SPIKE_PCT = 5.0
REFERENCE = ["Onion", "Soyabean", "Wheat"]          # calibration rows, always shown
# The markets we actually modeled (archive spellings inside the window, before the 2025 rename).
REFERENCE_MARKETS = {"Onion": "Lasalgaon", "Soyabean": "Indore", "Wheat": "Kanpur (Grain)"}
NEWS_ALIASES = {"soyabean": ["soybean"]}              # known spelling variants
EXCLUDED_NEWS_CATEGORY = "business.international-business"   # CBOT wire noise (Phase 3)

# Thresholds, set from what the first three commodities showed (all adjustable).
MIN_DAYS = 1500              # onion 2275, soyabean 2430, wheat 2850 reported days
MIN_TOTAL_SPIKES = 100       # soyabean 177 worked; wheat 11 did not
MIN_TEST_SPIKES = 20         # soyabean had 63 in test; wheat had 3
NOISY_BASELINE_PCT = 5.0     # onion 6.68 (poor), soyabean 2.76 (good)
SPARSE_NEWS = 100            # soyabean had 69 headlines and weak sentiment
LONG_GAP_DAYS = 30


def window_files(parquet_dir: Path):
    lo, hi = int(WINDOW_START[:4]), int(WINDOW_END[:4])
    files = [f for f in sorted(parquet_dir.glob("*.parquet")) if f.stem.isdigit() and lo <= int(f.stem) <= hi]
    if not files:
        raise SystemExit(f"No yearly parquet files for {lo}-{hi} found in {parquet_dir}")
    return files


def _clean(d: pd.DataFrame) -> pd.DataFrame:
    d = d[(d["Arrival_Date"] >= WINDOW_START) & (d["Arrival_Date"] <= WINDOW_END)].copy()
    d["Commodity"] = d["Commodity"].str.strip()
    d["Market"] = d["Market"].str.strip()
    return d


def pass1_market_days(files) -> pd.Series:
    """Distinct reported days per (Commodity, Market) in the window. Years are disjoint
    in dates, so summing per-year distinct counts is exact."""
    total = None
    for f in files:
        d = _clean(pd.read_parquet(f, columns=["Commodity", "Market", "Arrival_Date"]))
        g = d.groupby(["Commodity", "Market"])["Arrival_Date"].nunique()
        total = g if total is None else total.add(g, fill_value=0)
    return total.astype(int)


def fetch_live_keys(commodities) -> dict:
    """{commodity: set of formatting-insensitive market keys present in today's live feed}."""
    import live_pipeline as lp          # lazy: only needed for --live-only
    if not lp.API_KEY:
        raise SystemExit("DATA_GOV_IN_API_KEY missing - add it to .env before using --live-only.")
    keys = {}
    for c in commodities:
        try:
            df, truncated = lp.fetch_nationwide(c)
        except lp.FetchError as e:
            print(f"{c}: live feed unreachable ({e}) - skipped")
            continue
        keys[c] = {lp.market_key(m) for m in df["market"]} if len(df) else set()
        d = df["arrival_date"].max() if len(df) else None
        print(f"{c}: {len(keys[c])} markets / {len(df)} rows in today's live feed"
              f"{' (dated ' + str(d.date()) + ')' if d is not None and pd.notna(d) else ''}"
              f"{' - TRUNCATED' if truncated else ''}")
    return keys


def pick_pairs(market_days: pd.Series, names=None, top_markets=1, include_markets=None, live_keys=None) -> pd.DataFrame:
    """(Commodity, Market) pairs to evaluate. References use their pinned market."""
    df = market_days.rename("days").reset_index()
    include = [m.lower() for m in (include_markets or [])]
    ref_lookup = {r.lower(): r for r in REFERENCE}
    parts = []
    for commodity, g in df.groupby("Commodity"):
        g = g.sort_values("days", ascending=False)
        if commodity.lower() in ref_lookup:
            fixed = REFERENCE_MARKETS.get(ref_lookup[commodity.lower()])
            pick = g[g["Market"] == fixed].head(1)
            pick = pick if len(pick) else g.head(1)
            parts.append(pick.assign(reference=True, rank=1))
            continue
        wanted = any(n.lower() in commodity.lower() for n in names) if names else g["days"].iloc[0] >= MIN_DAYS
        if not wanted:
            continue
        if live_keys is not None:
            import live_pipeline as lp
            present = live_keys.get(commodity, set())
            top = g[(g["days"] >= MIN_DAYS) & g["Market"].map(lp.market_key).isin(present)]
            if top.empty:
                continue
            parts.append(top.assign(reference=False, rank=range(1, len(top) + 1)))
            continue
        top = g.head(top_markets)
        if include:
            extra = g[g["Market"].str.lower().apply(lambda m: any(i in m for i in include))]
            top = pd.concat([top, extra]).drop_duplicates("Market").sort_values("days", ascending=False)
        top = top.assign(reference=False, rank=range(1, len(top) + 1))
        parts.append(top)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].assign(reference=False, rank=1)


def pass2_series(files, pairs: pd.DataFrame) -> dict:
    """One daily price per (commodity, market): mean Modal_Price across varieties (the
    Phase 3 aggregation). Non-positive prices are dropped."""
    parts = []
    key = pairs[["Commodity", "Market"]].drop_duplicates()
    for f in files:
        d = _clean(pd.read_parquet(f, columns=["Commodity", "Market", "Arrival_Date", "Modal_Price"]))
        d = d.merge(key, on=["Commodity", "Market"])
        d["Modal_Price"] = pd.to_numeric(d["Modal_Price"], errors="coerce")
        d = d[d["Modal_Price"] > 0]
        parts.append(d.groupby(["Commodity", "Market", "Arrival_Date"])["Modal_Price"].mean().reset_index())
    allp = pd.concat(parts, ignore_index=True)
    allp["date"] = pd.to_datetime(allp["Arrival_Date"])
    return {k: g.sort_values("date")[["date", "Modal_Price"]].rename(columns={"Modal_Price": "price"}).reset_index(drop=True)
            for k, g in allp.groupby(["Commodity", "Market"])}


def series_metrics(s: pd.DataFrame) -> dict:
    n = len(s)
    nxt_pct = (s["price"].shift(-1) - s["price"]) / s["price"] * 100
    valid = nxt_pct.notna()
    spike = (nxt_pct >= SPIKE_PCT) & valid
    sv = spike[valid].reset_index(drop=True)
    nv = len(sv)

    def test_spikes(frac):
        return int(sv.iloc[int(nv * frac):].sum())

    gaps = s["date"].diff().dt.days.dropna()
    back = s["price"].pct_change() * 100
    calm = back[valid & ~spike].abs()
    return {
        "days": n, "first": s["date"].iloc[0].date(), "last": s["date"].iloc[-1].date(),
        "median_gap_d": float(gaps.median()) if len(gaps) else 0.0,
        "max_gap_d": int(gaps.max()) if len(gaps) else 0,
        "spikes": int(sv.sum()), "spike_rate_pct": round(100 * sv.sum() / nv, 2) if nv else 0.0,
        "test_spikes_80_20": test_spikes(0.8), "test_spikes_70_30": test_spikes(0.7),
        "calm_move_pct": round(float(calm.mean()), 2) if len(calm) else float("nan"),
    }


def news_counts(news: pd.DataFrame, commodity: str) -> dict:
    base = re.split(r"[(\[]", commodity)[0].strip().lower()
    forms = [base] + NEWS_ALIASES.get(base, [])
    pat = r"\b(?:" + "|".join(re.escape(f) for f in forms) + r")(?:s|es)?\b"
    hit = news[news["headline_text"].str.lower().str.contains(pat, na=False, regex=True)]
    return {"headlines": len(hit), "news_days": int(hit["publish_date"].nunique())}


def verdict(m: dict) -> tuple:
    test = m["test_spikes_80_20"] if m["test_spikes_80_20"] >= 5 else max(m["test_spikes_80_20"], m["test_spikes_70_30"])
    noisy = m["calm_move_pct"] == m["calm_move_pct"] and m["calm_move_pct"] > NOISY_BASELINE_PCT
    flags = []
    if m["max_gap_d"] > LONG_GAP_DAYS:
        flags.append("long-gaps")
    if m.get("headlines") is not None and m["headlines"] < SPARSE_NEWS:
        flags.append("sparse-news")
    if m["days"] < MIN_DAYS:
        v = "TOO-SPARSE"
    elif m["spikes"] < MIN_TOTAL_SPIKES or test < MIN_TEST_SPIKES:
        v = "TOO-FEW-SPIKES"
    elif noisy:
        v = "NOISY"
    else:
        v = "CANDIDATE"
    return v, ",".join(flags)


TIER_ORDER = {"CANDIDATE": 0, "NOISY": 1, "TOO-FEW-SPIKES": 2, "TOO-SPARSE": 3}


def screen(names=None, with_news=True, top_markets=1, include_markets=None, live_only=False,
           parquet_dir=PARQUET_DIR, news_path=NEWS_PATH) -> pd.DataFrame:
    files = window_files(parquet_dir)
    print(f"Pass 1/2: counting reported days per market ({len(files)} yearly files, {WINDOW_START}..{WINDOW_END})")
    market_days = pass1_market_days(files)
    live_keys = None
    if live_only:
        if not names:
            raise SystemExit("--live-only needs --names (it fetches the live feed once per matched commodity).")
        matched = [c for c in market_days.index.get_level_values(0).unique()
                   if any(n.lower() in c.lower() for n in names) and c.lower() not in [r.lower() for r in REFERENCE]]
        live_keys = fetch_live_keys(matched)
    pairs = pick_pairs(market_days, names, top_markets, include_markets, live_keys)
    print(f"Pass 2/2: building daily series for {len(pairs)} commodity/market pairs")
    series = pass2_series(files, pairs)

    rows = []
    for _, p in pairs.iterrows():
        k = (p["Commodity"], p["Market"])
        if k in series:
            rows.append({"commodity": p["Commodity"], "market": p["Market"], "rank": int(p["rank"]),
                         "reference": bool(p["reference"]), **series_metrics(series[k])})
    out = pd.DataFrame(rows)
    out["headlines"] = None
    out["news_days"] = None

    if with_news and len(out):
        if not Path(news_path).exists():
            print(f"(news file not found at {news_path} - skipping headline check)")
        else:
            news = pd.read_csv(news_path, usecols=["publish_date", "headline_category", "headline_text"])
            news["publish_date"] = pd.to_datetime(news["publish_date"], format="%Y%m%d", errors="coerce")
            news = news[(news["publish_date"] >= WINDOW_START) & (news["publish_date"] <= WINDOW_END)
                        & (news["headline_category"] != EXCLUDED_NEWS_CATEGORY)]
            # headlines depend on the commodity, not the market; check where any row is a CANDIDATE (+ references)
            pre = out.apply(lambda r: verdict({**r.to_dict(), "headlines": None})[0] == "CANDIDATE" or r["reference"], axis=1)
            counts = {c: news_counts(news, c) for c in out.loc[pre, "commodity"].unique()}
            out["headlines"] = out["commodity"].map(lambda c: counts.get(c, {}).get("headlines"))
            out["news_days"] = out["commodity"].map(lambda c: counts.get(c, {}).get("news_days"))

    vf = out.apply(lambda r: verdict({**r.to_dict(), "headlines": r["headlines"] if pd.notna(r["headlines"]) else None}), axis=1)
    out["verdict"] = [v[0] for v in vf]
    out["flags"] = [v[1] for v in vf]
    if top_markets > 1 or include_markets or live_only:
        out = out.sort_values(["commodity", "rank"]).reset_index(drop=True)
    else:
        out = out.sort_values(["verdict", "spikes"], key=lambda c: c.map(TIER_ORDER) if c.name == "verdict" else -c).reset_index(drop=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", help="commodity names (case-insensitive substring) to screen")
    ap.add_argument("--top-markets", type=int, default=1, help="compare the N best-covered markets per commodity")
    ap.add_argument("--include-markets", nargs="*", help="also evaluate markets whose name contains these words")
    ap.add_argument("--live-only", action="store_true", help="only markets present in today's live feed (needs --names)")
    ap.add_argument("--no-news", action="store_true")
    a = ap.parse_args(argv)
    res = screen(a.names, with_news=not a.no_news, top_markets=a.top_markets, include_markets=a.include_markets, live_only=a.live_only)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_PATH, index=False)
    show = res[["commodity", "rank", "market", "days", "spikes", "spike_rate_pct", "test_spikes_80_20",
                "calm_move_pct", "max_gap_d", "headlines", "verdict", "flags"]].copy()
    show.insert(1, "ref", res["reference"].map({True: "*", False: ""}))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + show.to_string(index=False))
    print("\n* = one of the original three, pinned to the markets we modeled (Lasalgaon / Indore / Kanpur (Grain)).")
    print(f"Thresholds: >= {MIN_DAYS} reported days, >= {MIN_TOTAL_SPIKES} spikes, >= {MIN_TEST_SPIKES} spikes in the "
          f"80/20 test period; NOISY if calm days move > {NOISY_BASELINE_PCT}% on average.")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()