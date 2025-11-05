import requests
import pandas as pd

fifa_url = "https://inside.fifa.com/api/ranking-overview"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 "
        "Safari/537.36"}

def fetch_overview(date_id: str | None = None, locale="en") -> dict:
    params = {"locale": locale, "rankingType": "football"}
    # every date has its own id
    # we will use the latest one
    if date_id:
        params["dateId"] = date_id  
    r = requests.get(fifa_url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_ranking_for(date_id: str) -> pd.DataFrame:
    j = fetch_overview(date_id)
    items = j.get("rankings", [])
    rows = []
    for it in items:
        r = it.get("rankingItem", {})
        tag = it.get("tag", {})
        name = r.get("name")
        rows.append({
            "team": name,
            "team_code": r.get("code") or r.get("countryCode"),
            "rank": r.get("rank"),
            "prev_rank": r.get("previousRank"),
            "points": r.get("totalPoints") or it.get("totalPoints"),
            "confed": tag.get("text") or tag.get("code"),
        })
    df = pd.DataFrame(rows)
    return df

def fetch_latest():
    df = fetch_ranking_for("id14898")  # latest id 
    df.to_csv("../data/fifa_latest.csv", index=False)
    print(df.head(10))