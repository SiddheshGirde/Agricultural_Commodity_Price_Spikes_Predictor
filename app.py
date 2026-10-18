"""
Phase 14/16 - Streamlit dashboard.

TWO TABS, deliberately kept separate:
  Historical backtest : the validated 2013-07-01..2023-06-30 window (news +
                        prices, full model, SHAP). Frozen by design: the news
                        dataset ends 2023-06-30.
  Live monitor        : prices collected daily by src/live_pipeline.py. The
                        dashboard NEVER calls the API - it only reads files the
                        collector already saved, so it works even when the API
                        is down. Live and historical series are NOT joined: the
                        gap between them would make rolling features meaningless.

Wheat has NO viable spike model (Phase 11: 11 events in 10 years). It is shown
transparently in both tabs instead of being hidden.
"""

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(os.environ.get("PBL_ROOT", r"C:\Users\siddh\Desktop\PBL"))
os.environ.setdefault("PBL_ROOT", str(BASE_DIR))   # live_pipeline reads the same root
sys.path.insert(0, str(BASE_DIR / "src"))
try:
    import live_pipeline as lp
    LIVE_IMPORT_ERROR = None
except Exception as e:                              # the historical tab must still work
    lp = None
    LIVE_IMPORT_ERROR = e

DATA_PATH = BASE_DIR / "data" / "interim" / "price_sentiment_with_target.csv"
NEWS_PATH = BASE_DIR / "data" / "interim" / "news_with_sentiment.csv"
MODELS_DIR = BASE_DIR / "models"
SHAP_DIR = BASE_DIR / "reports" / "shap"
METADATA_PATH = MODELS_DIR / "training_metadata.json"

NON_FEATURE_COLS = {
    "commodity", "date", "dominant_sentiment",
    "next_price", "next_date", "pct_change_to_next", "days_to_next_report",
    "spike",
}
COMMODITIES_WITH_MODEL = ["Onion", "Soyabean"]
ALL_COMMODITIES = ["Onion", "Soyabean", "Wheat"]
SPIKE_THRESHOLD_PCT = 5.0

STATUS_TEXT = {
    "ok": "Collected",
    "market_not_reporting": "Benchmark market did not report today (day closed)",
    "market_not_reported_yet": "Benchmark market has not reported yet - will check again this evening",
    "deferred_partial_day": "Today's data may be incomplete - will be saved in the evening run",
    "date_already_recorded": "Date already recorded",
    "api_unreachable": "Data source unreachable - will retry",
    "inconclusive_truncated": "Inconclusive (partial API response) - will retry",
    "no_data_nationwide": "API returned no data - will retry",
}


@st.cache_data
def load_price_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values(["commodity", "date"]).reset_index(drop=True)


@st.cache_data
def load_news_data():
    df = pd.read_csv(NEWS_PATH, parse_dates=["publish_date"])
    return df.sort_values("publish_date", ascending=False).reset_index(drop=True)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_model(commodity: str):
    return joblib.load(MODELS_DIR / f"{commodity.lower()}_random_forest.pkl")


def confidence_label(p: float) -> str:
    # Descriptive bands only - not a separate model output.
    if p < 0.20:
        return "Confident - low risk"
    if p < 0.40:
        return "Leaning no spike"
    if p < 0.60:
        return "Borderline - treat as uncertain"
    if p < 0.80:
        return "Leaning spike"
    return "Confident - high risk"


def show_performance(commodity: str, metadata: dict):
    mm = metadata[commodity]["models"]["random_forest"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{mm['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{mm['precision']*100:.1f}%")
    m3.metric("Recall", f"{mm['recall']*100:.1f}%")
    m4.metric("F1 Score", f"{mm['f1']*100:.1f}%")
    m5.metric("ROC-AUC", f"{mm['roc_auc']*100:.1f}%" if mm["roc_auc"] else "N/A")
    st.caption(
        f"Held-out test: {metadata[commodity]['test_rows']} days "
        f"({metadata[commodity]['test_date_range'][0]} to {metadata[commodity]['test_date_range'][1]}), "
        f"{metadata[commodity]['test_positive']} actual spike days. Precision/recall are modest, "
        f"not high-confidence - see the project report."
    )
    return mm


# ====================================================================== historical

def render_historical(commodity, df, news_df, metadata, feature_cols):
    st.warning(
        "**Backtest data:** bounded to the validated modeling period (2013-07-01 to 2023-06-30) - the news "
        "dataset ends there. 'Latest' below means the last date in that window, not today. "
        "See the **Live monitor** tab for prices collected daily."
    )
    sub = df[df["commodity"] == commodity].sort_values("date").reset_index(drop=True)
    latest_row = sub.iloc[-1]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{commodity} - Price History (backtest window)")
        st.line_chart(sub.set_index("date")["price"])
        st.metric(f"Last price in window ({latest_row['date'].date()})", f"Rs {latest_row['price']:.2f} / quintal")
    with col2:
        st.subheader("Recent News & Sentiment")
        recent_news = news_df[news_df["commodity"] == commodity].head(5)
        if len(recent_news) == 0:
            st.info("No matching headlines found for this commodity.")
        for _, row in recent_news.iterrows():
            emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(row["sentiment_label"], "")
            st.write(f"{emoji} **{row['publish_date'].date()}** - {row['headline_text']} "
                     f"_(model: {row['sentiment_label']})_")

    st.divider()
    st.subheader("Spike Prediction (backtest window)")

    if commodity not in COMMODITIES_WITH_MODEL:
        st.error(
            f"**No viable prediction model for {commodity}.** Only 11 spike events (5% next-day threshold) "
            f"occurred across {commodity}'s entire 10-year validated price history - too few to train or "
            f"evaluate a classifier meaningfully (the held-out test set contained just 3 positive examples). "
            f"All three baseline models tested (Logistic Regression, Random Forest, XGBoost) predicted zero "
            f"spikes. Reported as a genuine finding about {commodity}'s volatility at this market and "
            f"threshold, not a missing feature."
        )
        return

    model = load_model(commodity)
    X_latest = sub[feature_cols].iloc[[-1]]
    if X_latest.isnull().any(axis=1).iloc[0]:
        st.error("Latest row has missing feature values - cannot generate a prediction.")
        return

    proba = model.predict_proba(X_latest)[0][1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted spike probability", f"{proba*100:.1f}%")
    c2.metric("Prediction", "SPIKE" if proba >= 0.5 else "NO SPIKE")
    c3.metric("Confidence", confidence_label(proba))
    st.caption(f"Based on features from {latest_row['date'].date()} (last date in the backtest window), "
               f"predicting whether {commodity} rises {SPIKE_THRESHOLD_PCT:.0f}%+ on the next trading day. "
               f"The confidence band is descriptive only - see Model Performance below.")

    st.subheader("Model Performance (held-out test set, Phase 10/11)")
    mm = show_performance(commodity, metadata)
    st.write("**Confusion Matrix** (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(mm["confusion_matrix"],
                              index=["Actual: No Spike", "Actual: Spike"],
                              columns=["Predicted: No Spike", "Predicted: Spike"]))

    st.subheader("Most Influential Features (SHAP)")
    shap_path = SHAP_DIR / f"{commodity.lower()}_shap_bar.png"
    if shap_path.exists():
        st.image(str(shap_path), caption=f"{commodity} - SHAP feature importance (Phase 13)")
    else:
        st.info("SHAP plot not found - run src/explain.py first.")


# ====================================================================== live

def render_live(commodity, metadata):
    st.info(
        "**How this tab works:** a scheduled collector (`src/live_pipeline.py collect`) saves one price per "
        "trading day from data.gov.in. This dashboard only reads those saved files and **never calls the API**, "
        "so it works even when the API is down. Prediction is **price-only** (no live news is collected)."
    )
    if lp is None:
        st.error(f"Live module could not be loaded: {LIVE_IMPORT_ERROR}")
        return

    market = lp.BENCHMARK_MARKETS_LIVE[commodity]
    hist = lp.load_live_history(commodity)
    n = len(hist)
    need = lp.MIN_ROWS
    st.subheader(f"{commodity} - Live collection ({market})")

    # --- collector status
    last_attempt, status_text = "never", "No collection attempts logged yet"
    log = lp.read_log()
    if len(log):
        log = log[log["commodity"] == commodity]
        if len(log):
            last = log.iloc[-1]
            last_attempt = str(last["timestamp"]).replace("T", " ")
            status_text = STATUS_TEXT.get(last["status"], str(last["status"]))

    m1, m2, m3 = st.columns(3)
    m1.metric("Live trading days collected", f"{n}")
    if n:
        latest = hist.iloc[-1]
        m2.metric(f"Latest live price ({latest['Arrival_Date'].date()})", f"Rs {latest['price']:.2f} / quintal")
    else:
        m2.metric("Latest live price", "-")
    m3.metric("Last collector run", last_attempt)
    st.caption(f"Last collector result: **{status_text}**")

    st.progress(min(n / need, 1.0))
    st.caption(f"{min(n, need)} of {need} trading days collected - rolling features (7-day mean/std) need "
               f"{need} recent days before a prediction is meaningful.")

    if n >= 2:
        st.line_chart(hist.set_index("Arrival_Date")["price"])
    if n:
        shown = hist.sort_values("Arrival_Date", ascending=False).head(10)[
            ["Arrival_Date", "price", "min_price", "max_price", "n_varieties_reported"]].copy()
        shown["Arrival_Date"] = shown["Arrival_Date"].dt.date
        st.dataframe(shown, hide_index=True)
    st.caption("Live prices are kept separate from the 2013-2023 series on purpose: the multi-year gap between "
               "them would make rolling/lag features meaningless.")

    # --- prediction
    st.divider()
    st.subheader("Live Prediction (price-only)")
    try:
        r = lp.score_one(commodity)
    except Exception as e:
        st.error(f"Could not score live data: {e}")
        return

    if r["status"] == "scored":
        p = r["proba"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted spike probability", f"{p*100:.1f}%")
        c2.metric("Prediction", "SPIKE" if p >= 0.5 else "NO SPIKE")
        c3.metric("Confidence", confidence_label(p))
        st.caption(f"For the trading day after {r['as_of'].date()}. Price features only: sentiment is not "
                   f"available live, so it is set to 'no news'.")
        st.warning(
            "**Read with care:** the model was trained on 2013-2023 prices and uses absolute price levels, so "
            "scoring current prices is out-of-time - treat the probability as indicative, not calibrated. "
            "Historical test performance below applies to the 2021-2023 backtest, not to this live reading."
        )
        show_performance(commodity, metadata)
    elif r["status"] == "no_model":
        st.error(f"**No prediction for {commodity}.** Its live price is tracked, but it has no viable spike "
                 f"model: only 11 spike events in 10 years of history (Phase 11).")
    else:
        st.info(f"**No live prediction yet** - {r['reason']}.")

    # --- transparency
    with st.expander("Collector log (last 10 attempts)"):
        log = lp.read_log()
        log = log[log["commodity"] == commodity].tail(10) if len(log) else log
        if len(log):
            st.dataframe(log[["timestamp", "run_date", "status", "detail"]].fillna(""), hide_index=True)
        else:
            st.write("No collector runs logged yet.")


# ====================================================================== page

st.set_page_config(page_title="Agricultural Commodity Price Spike Predictor", layout="wide")
st.title("Agricultural Commodity Price Spike Predictor")
st.caption("FinBERT news sentiment + historical mandi prices -> next-trading-day spike prediction. "
           "Semester project | ProsusAI/finbert + Random Forest.")

df = load_price_data()
news_df = load_news_data()
metadata = load_metadata()
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

commodity = st.selectbox("Select a commodity", ALL_COMMODITIES)
tab_hist, tab_live = st.tabs(["Historical backtest (2013-2023)", "Live monitor"])
with tab_hist:
    render_historical(commodity, df, news_df, metadata, feature_cols)
with tab_live:
    render_live(commodity, metadata)