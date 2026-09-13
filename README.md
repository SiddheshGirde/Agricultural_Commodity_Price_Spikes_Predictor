# Agricultural Commodity Price Spike Predictor

NLP-based sentiment analysis of Indian agricultural market news (FinBERT),
combined with historical mandi price data, to predict next-trading-day
price spikes in Onion, Wheat, and Soyabean.

**Academic semester project.** Not a trading tool — see Limitations.

---

## What this actually does

1. Pulls daily wholesale (mandi) prices for Onion (Lasalgaon), Wheat
   (Kanpur Grain), and Soyabean (Indore) — the benchmark market for each
   commodity, from India's Agmarknet dataset.
2. Pulls ~3.8M Indian news headlines (2001–2023), filters to the ~2,000
   that mention these three commodities.
3. Scores each headline's sentiment with FinBERT (`ProsusAI/finbert`),
   with a targeted correction for a confirmed domain-gap bug (see below).
4. Aggregates sentiment to one row per (commodity, date), merges it onto
   the price calendar, engineers rolling/momentum price features.
5. Labels each day: does price rise ≥5% on the *next* trading day?
6. Trains and compares Logistic Regression, Random Forest, and XGBoost
   per commodity, using a strict chronological (not random) train/test
   split.
7. Explains the winning model with SHAP.
8. Serves everything through a Streamlit dashboard.

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
```

`requirements.txt` includes `pyarrow` (Parquet support) and `xgboost`
in addition to the original list — both added mid-project when they
were actually needed, not anticipated upfront.

## Data — not included in this repo, download separately

Raw source files are excluded from version control (large, and — for
the news dataset — of uncertain redistribution license). To reproduce
from scratch:

| Data | Source | Place in |
|---|---|---|
| Prices | [Daily Market Prices of Commodity India (2001–2026)](https://www.kaggle.com/datasets/khandelwalmanas/daily-commodity-prices-india), Parquet folder | `Data/parquet/` |
| News | [India News Headlines Dataset](https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset) | `data/raw/news/` |

Everything downstream of these two downloads **is** committed
(`data/interim/*.csv`, `models/*.pkl`) — the full multi-million-row raw
files are not, but every cleaned/aggregated intermediate output is, so
the dashboard and models are usable without re-running the full
pipeline.

## Reproducing the pipeline

Run in order — each stage's output feeds the next:

```bash
python src/preprocessing.py         # Phase 3 - clean price + news, filter to 3 commodities/benchmark markets
python src/sentiment.py             # Phase 5 - FinBERT scoring + price-direction override
python src/aggregate_sentiment.py   # Phase 6 - headline-level -> daily sentiment
python src/merge_price_sentiment.py # Phase 7 - align price calendar with sentiment (has_news flag)
python src/features.py              # Phase 8 - rolling/momentum price features
python src/generate_target.py       # Phase 9 - spike label (t -> t+1, next TRADING day)
python src/train.py                 # Phase 10/11 - chronological split, LR/RF/XGBoost comparison
python src/explain.py               # Phase 13 - SHAP feature importance (Random Forest)
streamlit run app.py                # Phase 14 - dashboard
```

## Key design decisions (and why)

- **Trading-day, not calendar-day, semantics throughout.** "Next day"
  means the next day the benchmark market actually reported a price,
  not the next calendar date. Necessary because mandis don't report on
  Sundays/holidays — without this, weekend gaps would either create
  spurious "1-day" spikes or leak information across the gap.
- **One benchmark market per commodity**, not a national average:
  Lasalgaon (Onion), Indore (Soyabean), Kanpur Grain (Wheat) — matches
  how these prices are actually referenced in Indian financial press,
  and avoids the aggregation/weighting complexity of blending thousands
  of mandis with inconsistent reporting.
- **Modeling window: 2013-07-01 to 2023-06-30.** The intersection of
  Soyabean's price data start (2013-07-01, the true binding constraint
  — not Wheat's, which was mistakenly used in an earlier draft of this
  window) and the news dataset's actual end date (2023-06-30, discovered
  during validation — the Kaggle listing's "21 years" claim describes
  the *start* of coverage, not how current it is).
- **Sentiment aggregation**: mean positive/neutral/negative probability
  across a day's articles, dominant label (mode), dispersion (std dev,
  explicit 0 — not NaN — on single-article days), `has_news` flag.
  Zero-news days are NOT filled with a manufactured "neutral" score —
  sentiment columns stay `NaN` until feature engineering, where they're
  imputed to 0 specifically because sklearn's `RandomForestClassifier`
  can't train on `NaN` (XGBoost could have used it natively, but a
  shared feature set was kept for a fair 3-way model comparison).

## Known limitation, found and corrected: FinBERT domain gap

FinBERT (trained on the Financial PhraseBank — company/investor
statements) initially scored **58.6% of "price up" headlines as
positive** — e.g. *"Onion prices shoot up by 52% in a month"* scored
0.92 positive. In that training domain, rising prices/profits are good
news; for a consumer commodity, they're the opposite of what this
project needs to predict. "Price down" headlines were scored correctly
(89.9% negative), confirming this was a specific, directional gap, not
random noise.

**Fix applied**: a targeted regex identifies price-rise language
co-occurring with the word "price(s)" and swaps the positive/negative
probabilities for those 130 headlines (out of 1,898). Scope
deliberately narrow — output/export-volume language showing the same
ambiguity (e.g. *"Soyabean exports up by 50%"*) was identified but left
uncorrected, since export growth and output growth push commodity
prices in *opposite* directions, and a blanket rule would introduce a
new bug rather than fix this one. Affects ≤2 headlines in the full
dataset — noted, not chased further.

## Known limitation: Wheat has no viable model

At the 5% next-day threshold, Wheat had **11 spike events across its
entire 10-year validated history** (0.39% base rate, vs. 22.5% for
Onion and 7.3% for Soyabean). A chronological test split puts only 3 of
those events in the held-out set — too few for precision/recall/F1 to
be statistically meaningful regardless of model or resampling
technique. All three baselines (Logistic Regression, Random Forest,
XGBoost) predicted zero spikes on Wheat.

**Decision**: keep the 5% threshold and 1-day horizon consistent across
all three commodities (no special-casing Wheat to get a better-looking
number), keep Wheat in the pipeline, and report this as a genuine
finding about Wheat's volatility profile at this market/threshold — not
a missing feature. The dashboard shows Wheat's price/news normally and
an explicit "no viable model" message instead of a fabricated
prediction.

## Result: sentiment mattered less than expected — also a real finding

Random Forest's SHAP importance ranking is dominated by price-derived
features (`rolling_std_7d`, `momentum_7d`, `pct_change_1d`, etc.) for
both Onion and Soyabean. `sentiment_negative_mean` barely enters
Onion's top 10; no sentiment feature appears in Soyabean's top 10 at
all. The FinBERT pipeline was built and validated carefully (including
finding and fixing the domain-gap bug above), and the honest conclusion
is that **it contributes little incremental predictive value over price
data alone** in this project's final models — plausibly because
(a) Onion's real crisis coverage is reactive/coincident with price
moves rather than leading them, and (b) Soyabean's sentiment coverage
is only ~1.7% of trading days. Reported as a finding, not hidden.

## Why Onion underperformed Soyabean despite better data

Counterintuitively, Random Forest scored far better on Soyabean
(F1 0.657, ROC-AUC 0.873) than Onion (F1 0.309, ROC-AUC 0.585), despite
Onion having more news coverage and a higher spike rate. Investigated
directly: Onion's *non-spike* days already average a 6.68% absolute
daily price move — a 5% "spike" barely stands out from routine noise.
Soyabean's non-spike days average only 2.76% — a real spike is a
genuine outlier against a much calmer baseline. Soyabean's spikes are
also more directly foreshadowed by the immediately preceding day's
price move (correlation r=-0.226, vs. Onion's r=-0.072) — confirmed
independently by SHAP, where `pct_change_1d` is Soyabean's single most
important feature.

## Other limitations

- **Headline-only news text** (no article body) — a defensible choice
  for FinBERT specifically (closer to its training distribution than
  full articles), not a workaround for lack of effort.
- **Day-level dates only**, both price and news sources — no intra-day
  timestamps, so same-day ordering between a headline's publish time
  and a price report is not resolvable with this data.
- **XGBoost's `scale_pos_weight` was computed with the standard formula
  and not tuned** — its true ceiling on this data is unknown, only its
  out-of-the-box performance was evaluated.
- **No live data feed.** All dashboard output reflects the fixed
  2013-07-01–2023-06-30 window, not real-time prices or news.

## Project structure


## Phase 16 (in progress): live data access — proof of concept only

`src/fetch_live_price.py` connects to the same government source
(data.gov.in's Agmarknet mandi price API, resource ID
`9ef84268-d588-465a-a308-a864a43d0070`) that the historical training
data was built from, confirming the fetch mechanism works end-to-end
against real, current data — not a static snapshot.

**What was found and fixed along the way** (real engineering, not a
clean first pass):
- The API silently returns an empty HTTP 502 for requests without a
  browser-like `User-Agent` header — confirmed by isolating headers as
  the only variable between an identical failing and succeeding
  request. Not documented anywhere; found by elimination.
- The shared public sample API key caps responses at 10 records
  regardless of the `limit` parameter requested — masked as an
  apparent "no data" result until traced back to the key itself.
- The `filters[market]` query parameter is silently ignored
  server-side — confirmed by sending two different market names and
  getting two different, mutually inconsistent result sets, neither
  matching either requested value. Workaround: fetch by commodity
  nationwide, filter to the target market client-side in pandas.
- All three benchmark markets (Lasalgaon, Indore, Kanpur Grain) now
  report under an "APMC"-suffixed name in live data, consistent with
  the same rename found in the historical bulk export around
  November 2025 (see Phase 3 validation).

**What this does NOT yet do**: fetching a single live price is not the
same as generating a live prediction. The trained models require
rolling/momentum features computed from a *sequence* of recent prices,
not one isolated day's value — appending a fetched live price onto the
historical series and regenerating that day's feature row was
deliberately left as **future work**, not built. The dashboard's
predictions remain based on the fixed 2013-07-01–2023-06-30 window.

**Also observed, not yet conclusive**: on the one day tested so far
(a Sunday), none of the three benchmark markets reported any price at
all, while hundreds of other markets nationwide did report that same
commodity. Consistent with mandis not trading on Sundays — the same
weekly reporting-gap pattern the historical data's trading-day-vs-
calendar-day target definition (Phase 9) was built to handle — but
confirmed only for one day so far. `data/interim/live_fetch_log.csv`
accumulates one row per market per run, for anyone who wants to keep
tracking this pattern across more days.