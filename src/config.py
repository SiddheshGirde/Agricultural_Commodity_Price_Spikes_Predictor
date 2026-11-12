"""
Single source of truth for which commodities the project covers.

Adding a commodity = one entry below, then re-run the pipeline stages (preprocessing ->
sentiment -> aggregate -> merge -> features -> target -> train). It does NOT decide whether
the commodity is worth modeling: run src/screen_commodities.py first.

Fields
  archive_name      exact commodity name in the bulk archive (also the name used everywhere downstream)
  archive_market    benchmark market as spelled in the archive inside the modeling window
  news_pattern      regex a headline must match to be tagged with this commodity
  news_exclude      optional regex; matching headlines are DROPPED and counted (never silently)
  live_market       benchmark market name in the LIVE feed (matched ignoring spacing/case/APMC)
  has_viable_model  set True only after training shows a usable model (enough test spikes, sensible
                    metrics). Controls whether the dashboard / live scoring offer a prediction.
  no_model_note     required when has_viable_model is False: the reason, shown on the dashboard
                    (state the real reason - 'too few spikes' and 'models did not beat chance' differ)
"""
import os
from pathlib import Path

# Repo root = the folder above src/. $PBL_ROOT overrides it (used for testing / unusual layouts).
PROJECT_ROOT = Path(os.environ.get("PBL_ROOT") or Path(__file__).resolve().parents[1])
PRICE_PARQUET_DIR = PROJECT_ROOT / "Data" / "parquet"
NEWS_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "news" / "india-news-headlines.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

MODELING_START = "2013-07-01"      # soyabean's price start was the binding constraint
MODELING_END = "2023-06-30"        # the news dataset ends here
EXCLUDED_NEWS_CATEGORY = "business.international-business"   # CBOT/global wire noise, not Indian mandi news

COMMODITIES = {
    "Onion": {
        "archive_name": "Onion", "archive_market": "Lasalgaon",
        "news_pattern": r"\bonion\b", "news_exclude": None,
        "live_market": "Lasalgaon APMC", "has_viable_model": True,
    },
    "Wheat": {
        "archive_name": "Wheat", "archive_market": "Kanpur (Grain)",
        "news_pattern": r"\bwheat\b", "news_exclude": None,
        "live_market": "Kanpur (Grain) APMC", "has_viable_model": False,
        "no_model_note": ("only 11 spike events (5% next-day threshold) in 10 years - too few to train or evaluate "
                          "a classifier. The held-out test set had 3 positives, and all three models predicted zero spikes."),
    },
    "Soyabean": {
        "archive_name": "Soyabean", "archive_market": "Indore",
        "news_pattern": r"\b(?:soybean|soyabean)\b", "news_exclude": None,
        "live_market": "Indore APMC", "has_viable_model": True,
    },
    "Potato": {
        "archive_name": "Potato", "archive_market": "Khatra",
        "news_pattern": r"\bpotato(?:es)?\b",
        "news_exclude": r"\b(?:sweet|couch|hot)\s+potato",   # different commodity / idioms
        "live_market": "Khatra APMC",
        # Trained and reviewed against a rule fixed in advance (ROC-AUC >= 0.65 and Random Forest precision
        # >= 3x the base rate). Failed both: best ROC-AUC 0.62; Random Forest flagged 46 days, caught 0 of 26.
        "has_viable_model": False,
        "no_model_note": ("it had enough spike events to train on (119, spread over the years), but none of the three "
                          "models beat chance usefully on the held-out test set (best ROC-AUC 0.62; Random Forest "
                          "flagged 46 days and caught none of the 26 test spikes)."),
    },
}

for _name, _cfg in COMMODITIES.items():
    assert _name == _cfg["archive_name"], f"config key '{_name}' must equal archive_name '{_cfg['archive_name']}'"
    assert _cfg["has_viable_model"] or _cfg.get("no_model_note"), f"{_name}: a commodity without a viable model needs a no_model_note"


def commodities_with_model():
    return [c for c, cfg in COMMODITIES.items() if cfg["has_viable_model"]]


def display_order():
    """Commodities with a viable model first, then the tracked-only ones (stable within each group)."""
    return sorted(COMMODITIES, key=lambda c: not COMMODITIES[c]["has_viable_model"])