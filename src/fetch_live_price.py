"""
Phase 16 - live price fetch from data.gov.in for the three benchmark
markets. Validated end-to-end: resource ID, required User-Agent header
(this endpoint returns an empty 502 without one), personal API key
(shared sample key caps at 10 records/request), date format (DD/MM/YYYY),
and current live market names (all three carry an "APMC" suffix not
present in the historical bulk export - see Phase 3 rename finding).

KNOWN API LIMITATION: filters[market] is silently ignored server-side
(confirmed - two different filter values returned two different,
mutually inconsistent result sets, neither matching the requested
market). Workaround: fetch by commodity only (large limit), filter to
the target market in pandas.

KNOWN DATA BEHAVIOR, not a bug: a benchmark market can have zero
records for a given day (confirmed on a Sunday - consistent with mandis
not trading/reporting on Sundays, the same trading-day-vs-calendar-day
gap this project's target definition was built to handle from Phase 9
onward). Callers must handle this as a normal case.
"""

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["DATA_GOV_IN_API_KEY"]
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

BENCHMARK_MARKETS_LIVE = {
    "Onion": "Lasalgaon APMC",
    "Soyabean": "Indore APMC",
    "Wheat": "Kanpur (Grain) APMC",
}

LIVE_LOG_PATH = Path(r"C:\Users\siddh\Desktop\PBL\data\interim\live_fetch_log.csv")


def fetch_commodity_nationwide(commodity: str, limit: int = 1000) -> pd.DataFrame:
    """filters[market] doesn't work server-side - always pulls the full
    nationwide day's records for one commodity."""
    params = {"api-key": API_KEY, "format": "json", "limit": limit, "filters[commodity]": commodity}
    r = requests.get(URL, params=params, headers=HEADERS)
    r.raise_for_status()  # surface real HTTP errors loudly rather than silently returning empty
    data = r.json()
    df = pd.DataFrame(data.get("records", []))
    if len(df):
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y", errors="coerce")
    return df


def fetch_benchmark_price(commodity: str, market: str) -> dict:
    """Returns a result dict either way - success or a documented gap,
    never raises just because a market didn't report today."""
    nationwide = fetch_commodity_nationwide(commodity)
    result = {
        "commodity": commodity, "market": market,
        "fetched_at": pd.Timestamp.now(),
        "nationwide_total": len(nationwide),
        "status": None, "modal_price": None, "arrival_date": None,
    }

    if len(nationwide) == 0:
        result["status"] = "no_data_nationwide"
        return result

    match = nationwide[nationwide["market"] == market]
    if len(match) == 0:
        result["status"] = "market_not_reporting_today"
        return result

    row = match.iloc[0]
    result["status"] = "ok"
    result["modal_price"] = row["modal_price"]
    result["arrival_date"] = row["arrival_date"]
    return result


if __name__ == "__main__":
    results = [fetch_benchmark_price(c, m) for c, m in BENCHMARK_MARKETS_LIVE.items()]

    for r in results:
        if r["status"] == "ok":
            print(f"{r['commodity']} @ {r['market']}: Rs {r['modal_price']} on {r['arrival_date'].date()} "
                  f"({r['nationwide_total']} reported nationwide today)")
        elif r["status"] == "market_not_reporting_today":
            print(f"{r['commodity']} @ {r['market']}: not reporting today "
                  f"({r['nationwide_total']} OTHER markets did) - normal gap, not an error")
        else:
            print(f"{r['commodity']} @ {r['market']}: zero {r['commodity']} records nationwide today - unusual, worth a second look if this repeats")

    # Append every run to a log - lets you see the actual gap pattern
    # (Sundays? specific holidays?) across many days, rather than one
    # anecdote, once you've run this a handful of times.
    log_df = pd.DataFrame(results)
    LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(LIVE_LOG_PATH, mode="a", header=not LIVE_LOG_PATH.exists(), index=False)
    print(f"\nLogged to {LIVE_LOG_PATH}")