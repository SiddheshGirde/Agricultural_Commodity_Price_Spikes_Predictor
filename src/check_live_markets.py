"""
READ-ONLY check: is a candidate (commodity, market) present in today's LIVE feed, and under
what name? The live feed formats market names differently from the bulk archive
('Kanpur(Grain) APMC', 'APMC Lasalgaon ', trailing spaces), so this matches with the same
formatting-insensitive key the collector uses and lists similarly named markets.

Run it in the evening (the day's feed fills in gradually) and on several days: a market
that appears one evening may not appear the next. Every run is appended to
data/interim/live_market_check.csv so the history shows how often each one reports.

USAGE
  python src/check_live_markets.py Tomato:Kharar Potato:Kharar Potato:Khatra "Potato:Bishnupur (Bankura)"
Nothing is saved to the price history and no day is closed.
"""
import sys
from datetime import datetime

import pandas as pd

import live_pipeline as lp

CHECK_LOG = lp.INTERIM / "live_market_check.csv"


def parse_pairs(args):
    pairs = []
    for a in args:
        if ":" not in a:
            raise SystemExit(f"Expected Commodity:Market, got '{a}'")
        c, m = a.split(":", 1)
        pairs.append((c.strip(), m.strip()))
    return pairs


def check(pairs):
    feeds, results = {}, []
    for commodity, market in pairs:
        if commodity not in feeds:
            try:
                feeds[commodity] = lp.fetch_nationwide(commodity)
            except lp.FetchError as e:
                feeds[commodity] = e
        got = feeds[commodity]
        res = {"timestamp": datetime.now().isoformat(timespec="seconds"), "commodity": commodity, "market": market,
               "found": False, "matched_names": "", "rows": 0, "feed_rows": 0, "feed_date": "", "note": ""}
        if isinstance(got, Exception):
            res["note"] = f"API unreachable ({got})"
        else:
            df, truncated = got
            res["feed_rows"] = len(df)
            if df.empty:
                res["note"] = "feed has no rows for this commodity name (check the spelling used by the API)"
            else:
                fd = df["arrival_date"].max()
                res["feed_date"] = str(fd.date()) if pd.notna(fd) else ""
                key = lp.market_key(market)
                hit = df[df["market"].map(lp.market_key) == key]
                if len(hit):
                    res.update(found=True, rows=len(hit), matched_names=" | ".join(sorted(hit["market"].unique())))
                    d = hit["arrival_date"].max()
                    res["note"] = f"latest row dated {d.date() if pd.notna(d) else '?'}"
                else:
                    word = market.split()[0]
                    near = sorted(df.loc[df["market"].str.contains(word, case=False, na=False, regex=False), "market"].unique())
                    res["note"] = f"not in feed; names containing '{word}': {near[:6] if near else 'none'}" + \
                                  ("; feed truncated - inconclusive" if truncated else "")
        results.append(res)
    return results


def main(argv):
    if not lp.API_KEY:
        raise SystemExit("DATA_GOV_IN_API_KEY missing - add it to .env first.")
    if not argv:
        raise SystemExit(__doc__)
    results = check(parse_pairs(argv))
    print(f"=== LIVE MARKET CHECK {datetime.now():%Y-%m-%d %H:%M} (read-only) ===")
    for r in results:
        tag = "FOUND    " if r["found"] else "NOT FOUND"
        extra = f" as '{r['matched_names']}' ({r['rows']} row(s))" if r["found"] else ""
        print(f"{tag} {r['commodity']}:{r['market']}{extra} - {r['note']} [feed: {r['feed_rows']} rows, dated {r['feed_date'] or '?'}]")
    lp.INTERIM.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(CHECK_LOG, mode="a", header=not CHECK_LOG.exists(), index=False)
    return results


if __name__ == "__main__":
    main(sys.argv[1:])