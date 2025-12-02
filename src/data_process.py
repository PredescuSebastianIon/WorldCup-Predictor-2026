#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

merge_csv= Path("../data/processed/merged_data.csv")
historical_rankings= Path("../data/fifa_historical.csv")
latest_rankings= Path("../data/fifa_latest.csv")

def merge():
    # loading source csvs
    historical_df = pd.read_csv(historical_rankings)
    latest_df = pd.read_csv(latest_rankings)

    # normalizing column names
    historical_df = historical_df.rename(columns={
        "rank_date" : "date",
        "country_full" : "country",
        "country_abrv" : "abbreviation",
        "total_points" : "points",
        "confederation" : "confed",
        "previous_points" : "prev_points"
    })

    latest_df = latest_df.rename(columns={
        "team" : "country",
        "team_code" : "abbreviation",
    })

    # Give latest rows the latest date
    latest_df["date"]="2025-10-17"

    # We take points from the historical snapshot "2024-06-20" and map them by country
    # latest.prev_points = historical[2024-06-20].points
    history_2024 = (
        historical_df.loc[historical_df["date"].astype(str).str.startswith("2024-06-20"), ["country", "points"]]
        .rename(columns={"points": "prev_points"})
    )
    latest_df = latest_df.merge(history_2024, on="country", how="left")

    # prev_rank = rank - rank_change when rank_change exists; else default to current rank
    mask = historical_df["rank"].ne(historical_df["rank_change"])
    historical_df["prev_rank"] = np.where(
        mask,
        historical_df["rank"] - historical_df["rank_change"],
        historical_df["rank"] # we shall decide if the prev_rank for the first date shall be 0, NaN or historical_df["rank"] = same as the rank
    )

    # Will keep prev_points relevnat for the first date so we don't have a jump from 0
    mask = historical_df["prev_points"].eq(0)
    historical_df.loc[mask, "prev_points"] = historical_df.loc[mask, "points"]

    print(historical_df)   

    # Define the final column order
    headers_merge = ["date","country", "abbreviation", "rank", "prev_rank", "points", "prev_points", "confed"]
    pd.DataFrame(columns=headers_merge).to_csv(merge_csv, index=False)

    # Reindex both frames to the same schema
    hist_out   = historical_df.reindex(columns=headers_merge)
    latest_out = latest_df.reindex(columns=headers_merge)

    # Concatanate
    merged = pd.concat([hist_out, latest_out], ignore_index=True)

    # Enforce integer types for rank
    merged["rank"] = pd.to_numeric(merged["rank"], errors="coerce").astype("Int64")
    merged["prev_rank"] = pd.to_numeric(merged["prev_rank"], errors="coerce").astype("Int64")

    merged.to_csv(merge_csv, index=False)
    print(merged)
