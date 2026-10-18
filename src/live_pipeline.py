"""
Phase 16b (v2) - live inference pipeline, restructured for reliability.

WHY THIS WAS RESTRUCTURED
  v1 appended today's live price directly after the 2023-06-30 historical
  tail, so rolling/lag features were computed ACROSS A ~3-YEAR GAP - a
  silent, meaningless result. v2 never mixes the two: live scoring uses
  ONLY consecutive recent live rows, and refuses (with a stated reason)
  when it doesn't have them.

DESIGN: collecting and scoring are separate steps
  collect : fetch -> retry with backoff -> append to live history.
            May fail (the API is unreliable); failure is logged, never fatal.
  score   : reads ONLY local files. Works with the API completely down.
  Run `collect` once a day (e.g. Windows Task Scheduler). The dashboard
  and `score` never touch the network.

DAILY-ONCE, DATEWISE RULES
  - One settled fetch per day: at/after SETTLE_HOUR, once a day has a finished
    outcome (row saved, market confirmed not reporting, or date already
    recorded) further runs that day make NO API call. An outage does NOT lock
    the day, so it can be retried.
  - Before SETTLE_HOUR nothing closes the day and today's partial data is never
    saved - but a COMPLETED earlier day can still be saved (catch-up after a
    missed evening run). The evening run then still collects today.
  - Datewise, insert-only: rows are keyed by the data's own arrival date. A
    date already in the history is never overwritten; past dates never change.
  - `collect --force` bypasses the once-a-day lock (it still never overwrites
    an existing date).

USAGE
  python src/live_pipeline.py            # collect, then score
  python src/live_pipeline.py collect    # at most one real fetch per day
  python src/live_pipeline.py collect --force
  python src/live_pipeline.py score
  python src/live_pipeline.py status
  python src/live_pipeline.py probe      # read-only: log how full the feed is right now

EXIT CODES (for the scheduler's retry loop)
  0 = finished (or nothing to do);  2 = at least one commodity's day is still
  open because of an outage/inconclusive result - retry later;  1 = a crash.

SCORING GUARDS (all must pass, otherwise no prediction is made)
  - >= MIN_ROWS live rows for that commodity
  - no gap between the last MIN_ROWS rows longer than MAX_INTERNAL_GAP_DAYS
  - latest row no older than MAX_STALE_DAYS

KNOWN LIMITS (stated, not hidden)
  - No live news pipeline: sentiment is zeroed (has_news=0), so this is a
    PRICE-ONLY prediction. SHAP showed price features dominate anyway.
  - Models were trained on 2013-2023 prices and use absolute price levels
    (price, min_price, rolling_mean...) as features. Scoring 2026 data is
    out-of-time; treat probabilities as indicative, not calibrated.
  - Wheat has no viable model (Phase 11): its prices are collected, not scored.
"""

import csv
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(os.environ.get("PBL_ROOT", r"C:\Users\siddh\Desktop\PBL"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from features import engineer_price_features   # reused, not reimplemented
from train import NON_FEATURE_COLS, MODELS_DIR  # reused, not reimplemented

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.environ.get("DATA_GOV_IN_API_KEY")
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
# This endpoint returns an empty 502 for requests without a browser-like User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

INTERIM = PROJECT_ROOT / "data" / "interim"
LIVE_HISTORY_PATH = INTERIM / "live_price_history.csv"
# NOT live_fetch_log.csv: fetch_live_price.py writes that file with a different column layout.
LIVE_LOG_PATH = INTERIM / "live_collect_log.csv"
LIVE_RAW_DIR = INTERIM / "live_raw"
LABELED_PATH = INTERIM / "price_sentiment_with_target.csv"

BENCHMARK_MARKETS_LIVE = {
    "Onion": "Lasalgaon APMC",
    "Soyabean": "Indore APMC",
    "Wheat": "Kanpur (Grain) APMC",
}
COMMODITIES_WITH_MODEL = ["Onion", "Soyabean"]

RETRIES = 3
BACKOFF_SECONDS = [3, 8]       # waits between attempts 1->2 and 2->3
TIMEOUT = (10, 30)             # (connect, read) seconds - never wait forever
PAGE_LIMIT = 1000

# Before this local hour, today's feed may still be filling in (hourly refreshes were
# observed). Nothing can CLOSE a day before it, and today's partial data is never saved.
# 18 is a guess - adjust after looking at the collector log for a few days.
SETTLE_HOUR = 18

MIN_ROWS = 7                   # rolling_mean_7d / rolling_std_7d need 7 rows
MAX_INTERNAL_GAP_DAYS = 7      # mandis close on Sundays/holidays; a week is already generous
MAX_STALE_DAYS = 7

LOG_COLS = ["timestamp", "run_date", "settled", "commodity", "market", "status", "detail"]
HISTORY_COLS = ["Commodity", "Arrival_Date", "price", "min_price", "max_price",
                "n_varieties_reported", "source"]


# Outcomes that finish the day. api_unreachable / inconclusive_truncated / no_data_nationwide do NOT.
LOCK_STATUSES = {"ok", "market_not_reporting", "date_already_recorded"}


def _today():
    """Calendar date used for the once-a-day lock (patched in tests)."""
    return datetime.now().date()


def _hour():
    """Local hour of day (patched in tests)."""
    return datetime.now().hour


class FetchError(Exception):
    pass


# ---------------------------------------------------------------- collect

def fetch_nationwide(commodity: str):
    """Returns (dataframe, truncated). filters[market] is silently ignored
    server-side (confirmed Phase 16), so fetch by commodity and filter in
    pandas. `truncated` is True if the API reports more rows than it returned -
    in which case "market not found" is INCONCLUSIVE, not "not reporting"."""
    params = {"api-key": API_KEY, "format": "json", "limit": PAGE_LIMIT,
              "filters[commodity]": commodity}
    last_err = "unknown"
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                payload = r.json()
                records = payload.get("records", [])
                total = int(payload.get("total", len(records)))
                df = pd.DataFrame(records)
                if len(df):
                    df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y", errors="coerce")
                    for c in ("modal_price", "min_price", "max_price"):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                return df, total > len(records)
            last_err = f"HTTP {r.status_code}"
        except (requests.exceptions.RequestException, ValueError) as e:
            last_err = type(e).__name__
        if attempt < RETRIES:
            time.sleep(BACKOFF_SECONDS[attempt - 1])
    raise FetchError(last_err)


def market_key(name) -> str:
    """Formatting-insensitive market identity. The live feed formats names differently
    from the bulk archive: 'Kanpur(Grain) APMC' vs 'Kanpur (Grain) APMC', trailing
    spaces, and APMC as a prefix ('APMC Lasalgaon') or suffix ('Indore APMC') or absent.
    Drops whitespace, case and a leading/trailing 'APMC'. Sub-yards stay distinct:
    'Lasalgaon(Vinchur)' != 'Lasalgaon'."""
    k = re.sub(r"\s+", "", str(name)).lower()
    if k.startswith("apmc") and len(k) > 4:
        k = k[4:]
    if k.endswith("apmc") and len(k) > 4:
        k = k[:-4]
    return k


def aggregate_benchmark(df: pd.DataFrame, market: str, commodity: str):
    """Mean across varieties at the benchmark market - the same aggregation
    Phase 3 applied to historical data. Returns (row, matched_rows) or (None, None)."""
    if df.empty:
        return None, None
    match = df[(df["market"].map(market_key) == market_key(market)) & df["arrival_date"].notna()]
    if match.empty:
        return None, None
    # Never average prices from different dates into one row.
    match = match[match["arrival_date"] == match["arrival_date"].max()]
    row = {
        "Commodity": commodity,
        "Arrival_Date": match["arrival_date"].iloc[0],
        "price": match["modal_price"].mean(),
        "min_price": match["min_price"].mean(),
        "max_price": match["max_price"].mean(),
        "n_varieties_reported": match["variety"].nunique(),
        "source": "live",
    }
    return row, match


def load_live_history(commodity: str = None) -> pd.DataFrame:
    if not LIVE_HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLS)
    h = pd.read_csv(LIVE_HISTORY_PATH, parse_dates=["Arrival_Date"])
    if commodity is not None:
        h = h[h["Commodity"] == commodity]
    return h.sort_values("Arrival_Date").reset_index(drop=True)


def append_live_row(row: dict) -> bool:
    """Datewise, insert-only. Returns True if the row was added, False if that
    (Commodity, Arrival_Date) already exists - existing rows are never changed."""
    existing = load_live_history()
    if len(existing):
        dup = (existing["Commodity"] == row["Commodity"]) & (existing["Arrival_Date"] == row["Arrival_Date"])
        if dup.any():
            return False
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined = combined.sort_values(["Commodity", "Arrival_Date"])
    INTERIM.mkdir(parents=True, exist_ok=True)
    combined.to_csv(LIVE_HISTORY_PATH, index=False)
    return True


def read_log() -> pd.DataFrame:
    """Tolerant log reader. Earlier versions wrote 5- and 6-column rows into this
    same file, so a plain pd.read_csv can fail on a mixed file. Legacy rows carry no
    'settled' flag: it is reconstructed from the row's timestamp (a run before
    SETTLE_HOUR could not have closed the day), never assumed True."""
    if not LIVE_LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLS)
    out = []
    with open(LIVE_LOG_PATH, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd, None)                                   # header (any legacy layout)
        for r in rd:
            if len(r) == 7:
                ts, run_date, settled, commodity, market, status, detail = r
                settled = str(settled).strip().lower() == "true"
            elif len(r) == 6:
                ts, run_date, commodity, market, status, detail = r
                settled = None
            elif len(r) == 5:
                ts, commodity, market, status, detail = r
                run_date, settled = ts[:10], None
            else:
                continue
            if settled is None:
                t = pd.to_datetime(ts, errors="coerce")
                settled = bool(pd.notna(t) and t.hour >= SETTLE_HOUR)
            out.append([ts, run_date, settled, commodity, market, status, detail])
    return pd.DataFrame(out, columns=LOG_COLS)


def ensure_log_format():
    """One-time, automatic repair: if the file has a legacy header or mixed row
    widths, back it up (once) and rewrite it in the current 7-column layout."""
    if not LIVE_LOG_PATH.exists():
        return
    with open(LIVE_LOG_PATH, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        current = header == LOG_COLS and all(len(r) == 7 for r in rd if r)
    if current:
        return
    bak = LIVE_LOG_PATH.with_name(LIVE_LOG_PATH.name + ".bak")
    if not bak.exists():
        shutil.copy2(LIVE_LOG_PATH, bak)
    read_log().to_csv(LIVE_LOG_PATH, index=False)


def log_fetch(commodity: str, market: str, status: str, detail: str = "", settled: bool = True):
    ensure_log_format()
    entry = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"),
                           "run_date": str(_today()), "settled": settled,
                           "commodity": commodity, "market": market,
                           "status": status, "detail": detail}], columns=LOG_COLS)
    INTERIM.mkdir(parents=True, exist_ok=True)
    entry.to_csv(LIVE_LOG_PATH, mode="a", header=not LIVE_LOG_PATH.exists(), index=False)


def done_today(commodity: str):
    """Finishing status if today already has a finished AND settled outcome
    (logged at/after SETTLE_HOUR, or forced), else None. Earlier outcomes never
    close the day, so a morning/catch-up run cannot block the evening run."""
    log = read_log()
    hit = log[(log["commodity"] == commodity) & (log["run_date"] == str(_today()))
              & (log["status"].isin(LOCK_STATUSES)) & log["settled"]]
    return hit["status"].iloc[-1] if len(hit) else None


def near_matches(df: pd.DataFrame, market: str) -> list:
    """Markets in today's feed sharing the benchmark's first word (e.g. 'Lasalgaon') -
    shows in the log whether a market is absent or merely renamed."""
    key = market.split()[0]
    return sorted(df.loc[df["market"].str.contains(key, case=False, na=False, regex=False), "market"].unique())


def collect(force: bool = False):
    if not API_KEY:
        raise SystemExit("DATA_GOV_IN_API_KEY missing - add it to .env before running collect.")
    settled = force or _hour() >= SETTLE_HOUR
    print(f"=== COLLECT ({_today()}, {'settled' if settled else f'before {SETTLE_HOUR}:00 - today is not final yet'}) ===")
    left_open = []   # commodities whose day is still open because of a FAILURE (worth retrying)
    for commodity, market in BENCHMARK_MARKETS_LIVE.items():
        finished = done_today(commodity)
        if finished and not force:
            print(f"{commodity}: already collected today ({finished}) - skipping, no API call.")
            continue

        try:
            df, truncated = fetch_nationwide(commodity)
        except FetchError as e:
            print(f"{commodity}: API UNREACHABLE after {RETRIES} attempts ({e}). Nothing collected; "
                  f"today stays open - re-run later.")
            log_fetch(commodity, market, "api_unreachable", str(e), settled)
            left_open.append(commodity)
            continue

        row, matched = aggregate_benchmark(df, market, commodity)
        if row is None:
            if truncated:
                print(f"{commodity}: {market} not found, but the API returned fewer rows than its total - "
                      f"INCONCLUSIVE, today stays open.")
                log_fetch(commodity, market, "inconclusive_truncated", settled=settled)
                left_open.append(commodity)
            elif df.empty:
                print(f"{commodity}: API returned zero rows nationwide - today stays open.")
                log_fetch(commodity, market, "no_data_nationwide", settled=settled)
                left_open.append(commodity)
            else:
                feed_date = df["arrival_date"].max()
                near = near_matches(df, market)
                detail = (f"{len(df)} other markets reported (feed dated "
                          f"{feed_date.date() if pd.notna(feed_date) else '?'}); names containing "
                          f"'{market.split()[0]}': {near[:6] if near else 'none'}")
                if settled:
                    print(f"{commodity}: {market} did not report today - day closed. {detail}")
                    log_fetch(commodity, market, "market_not_reporting", detail, settled)
                else:
                    print(f"{commodity}: {market} has not reported yet - may still report later today; "
                          f"day stays open. {detail}")
                    log_fetch(commodity, market, "market_not_reported_yet", detail, settled)
            continue

        row_date = row["Arrival_Date"].date()
        if row_date >= _today() and not settled:
            print(f"{commodity}: {market} has data dated {row_date}, but today may still be incomplete - "
                  f"NOT saved; the evening run will save it.")
            log_fetch(commodity, market, "deferred_partial_day", str(row_date), settled)
            continue

        if not append_live_row(row):
            print(f"{commodity}: {row_date} is already recorded - left unchanged (insert-only).")
            log_fetch(commodity, market, "date_already_recorded", str(row_date), settled)
            continue

        LIVE_RAW_DIR.mkdir(parents=True, exist_ok=True)   # keep raw rows so aggregation can be redone later
        matched.to_csv(LIVE_RAW_DIR / f"{commodity.lower()}_{row_date}.csv", index=False)
        note = "" if settled else "  (earlier completed day; today stays open for the evening run)"
        print(f"{commodity}: OK - added {row_date}  price={row['price']:.2f} "
              f"({row['n_varieties_reported']} varieties){note}")
        log_fetch(commodity, market, "ok", f"price={row['price']:.2f}", settled)
    return left_open


PROBE_LOG_PATH = INTERIM / "live_feed_probe.csv"


def probe():
    """READ-ONLY. Fetches and records feed counts so SETTLE_HOUR can be chosen from
    evidence. Never saves a price and never closes a day. Run it at several hours."""
    if not API_KEY:
        raise SystemExit("DATA_GOV_IN_API_KEY missing - add it to .env before running probe.")
    print(f"=== PROBE {datetime.now():%Y-%m-%d %H:%M} (nothing is saved or locked) ===")
    out = []
    for commodity, market in BENCHMARK_MARKETS_LIVE.items():
        try:
            df, truncated = fetch_nationwide(commodity)
        except FetchError as e:
            print(f"{commodity}: API unreachable ({e})")
            continue
        mine = df[df["market"].map(market_key) == market_key(market)] if len(df) else df
        feed_date = df["arrival_date"].max() if len(df) else pd.NaT
        bench_date = mine["arrival_date"].max() if len(mine) else pd.NaT
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), "commodity": commodity,
                 "feed_rows": len(df), "feed_markets": int(df["market"].nunique()) if len(df) else 0,
                 "feed_date": feed_date.date() if pd.notna(feed_date) else "",
                 "benchmark_rows": len(mine),
                 "benchmark_date": bench_date.date() if pd.notna(bench_date) else "",
                 "truncated": truncated}
        out.append(entry)
        print(f"{commodity}: feed {entry['feed_rows']} rows / {entry['feed_markets']} markets "
              f"(dated {entry['feed_date'] or '?'}) | {market}: {entry['benchmark_rows']} row(s)"
              f"{' dated ' + str(entry['benchmark_date']) if entry['benchmark_date'] else ' - not in feed'}")
    if out:
        INTERIM.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out).to_csv(PROBE_LOG_PATH, mode="a", header=not PROBE_LOG_PATH.exists(), index=False)


# ------------------------------------------------------------------ score

def check_scoreable(hist: pd.DataFrame, today: pd.Timestamp):
    """Returns (ok, reason). Refuses rather than computing features over gaps."""
    if len(hist) < MIN_ROWS:
        return False, (f"only {len(hist)} live row(s) collected; need {MIN_ROWS} recent trading days "
                       f"before rolling features are meaningful")
    tail = hist.tail(MIN_ROWS)
    max_gap = tail["Arrival_Date"].diff().dt.days.dropna().max()
    if max_gap > MAX_INTERNAL_GAP_DAYS:
        return False, f"gap of {int(max_gap)} days inside the last {MIN_ROWS} rows (limit {MAX_INTERNAL_GAP_DAYS})"
    stale = (today - tail["Arrival_Date"].iloc[-1]).days
    if stale > MAX_STALE_DAYS:
        return False, f"latest live row is {stale} days old (limit {MAX_STALE_DAYS})"
    return True, ""


def build_scoreable_row(hist: pd.DataFrame, feature_cols: list):
    engineered = engineer_price_features(hist.rename(columns={"Arrival_Date": "date"}))
    latest = engineered.iloc[[-1]].copy()
    # No live news pipeline: explicit "no news" convention from Phase 7/8.
    for c in ("sentiment_positive_mean", "sentiment_neutral_mean",
              "sentiment_negative_mean", "sentiment_dispersion"):
        latest[c] = 0.0
    latest["article_count"] = 0
    latest["has_news"] = 0
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise ValueError(f"Feature columns missing - pipeline drift: {missing}")
    return latest[feature_cols], latest


def get_feature_cols() -> list:
    return [c for c in pd.read_csv(LABELED_PATH, nrows=1).columns if c not in NON_FEATURE_COLS]


def score_one(commodity: str, feature_cols: list = None, today: pd.Timestamp = None) -> dict:
    """Pure function (no printing, no network) used by both the CLI and the
    dashboard so they can never disagree. status is one of:
      scored | not_scoreable | no_model | missing_values"""
    today = today if today is not None else pd.Timestamp.now().normalize()
    feature_cols = feature_cols if feature_cols is not None else get_feature_cols()
    hist = load_live_history(commodity)
    out = {"commodity": commodity, "status": None, "reason": "", "proba": None,
           "as_of": hist["Arrival_Date"].max() if len(hist) else None,
           "rows": len(hist), "min_rows": MIN_ROWS}

    ok, reason = check_scoreable(hist, today)
    if not ok:
        out.update(status="not_scoreable", reason=reason)
        return out
    X, latest = build_scoreable_row(hist, feature_cols)
    out["as_of"] = latest["date"].iloc[0]
    if X.isnull().any(axis=1).iloc[0]:
        out.update(status="missing_values", reason="feature row contains missing values")
        return out
    if commodity not in COMMODITIES_WITH_MODEL:
        out.update(status="no_model", reason="no viable model for this commodity (Phase 11)")
        return out
    model = joblib.load(MODELS_DIR / f"{commodity.lower()}_random_forest.pkl")
    out.update(status="scored", proba=float(model.predict_proba(X)[0][1]))
    return out


def score(today: pd.Timestamp = None):
    today = today if today is not None else pd.Timestamp.now().normalize()
    feature_cols = get_feature_cols()
    results = {}
    print("\n=== SCORE (local files only - no network) ===")
    for commodity in BENCHMARK_MARKETS_LIVE:
        r = score_one(commodity, feature_cols, today)
        if r["status"] == "scored":
            label = "SPIKE" if r["proba"] >= 0.5 else "NO SPIKE"
            print(f"{commodity}: {label} (probability {r['proba']*100:.1f}%) for the trading day after "
                  f"{r['as_of'].date()}  [PRICE-ONLY: no live news, sentiment zeroed]")
            results[commodity] = r["proba"]
        else:
            tail = f" ({r['as_of'].date()})" if r["status"] == "no_model" and r["as_of"] is not None else ""
            print(f"{commodity}: NOT SCORING - {r['reason']}{tail}.")
            results[commodity] = None
    return results


def status():
    print("=== STATUS ===")
    for commodity in BENCHMARK_MARKETS_LIVE:
        h = load_live_history(commodity)
        last = h["Arrival_Date"].max().date() if len(h) else "-"
        d = done_today(commodity)
        print(f"{commodity}: {len(h)} live row(s), latest {last}  (need {MIN_ROWS} recent rows to score) | "
              f"today: {d if d else 'not yet collected'}")
    log = read_log()
    if len(log):
        print("\nLast 6 fetch attempts:")
        print(log.tail(6)[["timestamp", "commodity", "status", "settled", "detail"]].to_string(index=False))


EXIT_DONE, EXIT_RETRY = 0, 2


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else "run"
    if cmd == "collect":
        return EXIT_RETRY if collect(force="--force" in argv) else EXIT_DONE
    if cmd == "probe":
        probe()
        return EXIT_DONE
    if cmd == "score":
        score()
        return EXIT_DONE
    if cmd == "status":
        status()
        return EXIT_DONE
    if cmd == "run":
        left_open = collect(force="--force" in argv)
        score()
        return EXIT_RETRY if left_open else EXIT_DONE
    raise SystemExit("usage: live_pipeline.py [collect|score|status|probe|run] [--force]")


if __name__ == "__main__":
    sys.exit(main(sys.argv))