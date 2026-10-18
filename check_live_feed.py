import os, requests, pandas as pd
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["DATA_GOV_IN_API_KEY"]
URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# (name fragments to look for in market names, district to inspect)
CHECKS = {
    "Onion":    (["Lasalgaon", "Pimpalgaon", "Niphad", "Vinchur"], "Nashik"),
    "Soyabean": (["Indore"], "Indore"),
    "Wheat":    (["Kanpur"], "Kanpur"),
}

for commodity, (names, district) in CHECKS.items():
    r = requests.get(URL, headers=H, timeout=30,
                     params={"api-key": KEY, "format": "json", "limit": 1000, "filters[commodity]": commodity})
    data = r.json()
    df = pd.DataFrame(data.get("records", []))
    print(f"\n=== {commodity}: HTTP {r.status_code}, API total {data.get('total')}, returned {len(df)} ===")
    if df.empty:
        continue
    print("arrival dates in feed:", df["arrival_date"].value_counts().to_dict())
    print("distinct markets:", df["market"].nunique(), "| states:", df["state"].nunique())
    hit = df[df["market"].str.contains("|".join(names), case=False, na=False)]
    print(f"markets whose name contains {names}:", sorted(hit["market"].unique()) or "NONE")
    dist = df[df["district"].str.contains(district, case=False, na=False)]
    print(f"markets in district '{district}':", sorted(dist["market"].unique()) or "NONE")
    