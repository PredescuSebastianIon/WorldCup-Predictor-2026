#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

merge_csv= Path("../data/processed/merged_data.csv")
historical_rankings= Path("../data/fifa_historical.csv")
latest_rankings= Path("../data/fifa_latest.csv")

all_matches = Path("../data/all_matches.csv")
filtered_matches = Path("../data/processed/all_matches_filtered.csv")
enriched_matches = Path("../data/processed/all_matches_enriched.csv")

teams_master = Path("../data/teams_master.csv")
relevant_matches = Path("../data/processed/all_matches_relevant_teams.csv")


def merge():
    # make sure ../data/processed exists
    merge_csv.parent.mkdir(parents=True, exist_ok=True)

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

    merged["country"] = merged["country"].replace({
        "USA": "United States",
        "Côte d'Ivoire": "Ivory Coast",
        "Cabo Verde": "Cape Verde"
    })

    # Enforce integer types for rank
    merged["rank"] = pd.to_numeric(merged["rank"], errors="coerce").astype("Int64")
    merged["prev_rank"] = pd.to_numeric(merged["prev_rank"], errors="coerce").astype("Int64")

    merged.to_csv(merge_csv, index=False)
    print(merged)

def filter_all_matches(input_path: Path = all_matches,
                       output_path: Path = filtered_matches,
                       min_year: int = 1993) -> pd.DataFrame:
    """
    Filter all_matches.csv to keep only matches from min_year onwards.
    
    Parameters:
    -----------
    input_path : Path
        Path to input CSV file
    output_path : Path
        Path to save filtered CSV file
    min_year : int
        Minimum year to include (inclusive)
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    # make sure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # assume there is a 'date' column like in international_matches.csv
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter by year
    df = df[df["date"].dt.year >= min_year].copy()

    df.to_csv(output_path, index=False)
    print(f"Filtered matches saved to {output_path}")
    print(f"Shape: {df.shape}, Years: {df['date'].dt.year.min()} to {df['date'].dt.year.max()}")
    return df


def enrich_dataset(matches_path: Path = filtered_matches,
                   rankings_path: Path = merge_csv,
                   output_path: Path = enriched_matches) -> pd.DataFrame:
    """
    Enrich filtered matches with FIFA rankings from merged_data.csv
    (latest ranking on or before match date for home & away teams).
    
    Parameters:
    -----------
    matches_path : Path
        Path to filtered matches CSV
    rankings_path : Path
        Path to merged rankings CSV
    output_path : Path
        Path to save enriched CSV file
    
    Returns:
    --------
    pd.DataFrame
        Enriched DataFrame
    """
    # make sure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(matches_path)
    rankings = pd.read_csv(rankings_path)

    # parse dates
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    rankings["date"] = pd.to_datetime(rankings["date"], errors="coerce")

    # make sure team names are same type everywhere
    matches["home_team"] = matches["home_team"].astype(str)
    matches["away_team"] = matches["away_team"].astype(str)
    rankings["country"] = rankings["country"].astype(str)

    # Sort rankings by date for merge_asof
    rankings = rankings.sort_values("date")

    # ----------------- HOME SIDE -----------------
    home_rankings = rankings.rename(columns={
        "country": "home_team",
        "rank": "home_rank",
        "points": "home_points",
        "prev_rank": "home_prev_rank",
        "prev_points": "home_prev_points",
        "confed": "home_confed",
    })

    # Sort matches by date for merge_asof
    matches_sorted = matches.sort_values("date").reset_index(drop=True)
    home_rankings_sorted = home_rankings.sort_values("date").reset_index(drop=True)

    matches_home = pd.merge_asof(
        matches_sorted,
        home_rankings_sorted[["home_team", "date", "home_rank", "home_points",
                              "home_prev_rank", "home_prev_points", "home_confed"]],
        left_on="date",
        right_on="date",
        by="home_team",
        direction="backward",
    )

    # ----------------- AWAY SIDE -----------------
    away_rankings = rankings.rename(columns={
        "country": "away_team",
        "rank": "away_rank",
        "points": "away_points",
        "prev_rank": "away_prev_rank",
        "prev_points": "away_prev_points",
        "confed": "away_confed",
    })

    # Sort again by date for merge_asof
    matches_home_sorted = matches_home.sort_values("date").reset_index(drop=True)
    away_rankings_sorted = away_rankings.sort_values("date").reset_index(drop=True)

    matches_full = pd.merge_asof(
        matches_home_sorted,
        away_rankings_sorted[["away_team", "date", "away_rank", "away_points",
                              "away_prev_rank", "away_prev_points", "away_confed"]],
        left_on="date",
        right_on="date",
        by="away_team",
        direction="backward",
    )

    # dtypes & simple derived features
    for col in ["home_rank", "away_rank", "home_prev_rank", "away_prev_rank"]:
        matches_full[col] = pd.to_numeric(matches_full[col], errors="coerce").astype("Int64")

    for col in ["home_points", "away_points", "home_prev_points", "away_prev_points"]:
        matches_full[col] = pd.to_numeric(matches_full[col], errors="coerce")

    matches_full["rank_diff"] = matches_full["home_rank"] - matches_full["away_rank"]
    matches_full["points_diff"] = matches_full["home_points"] - matches_full["away_points"]
    matches_full["score_diff"] = matches_full["home_score"] - matches_full["away_score"]
    matches_full["same_confed"] = (matches_full["home_confed"] == matches_full["away_confed"]).astype("Int64")

    # Check for missing rankings
    missing_home = matches_full["home_rank"].isna().sum()
    missing_away = matches_full["away_rank"].isna().sum()
    if missing_home > 0 or missing_away > 0:
        print(f"Warning: {missing_home} home teams and {missing_away} away teams have missing rankings")

    matches_full.to_csv(output_path, index=False)
    print(f"Enriched matches saved to {output_path}")
    print(f"Shape: {matches_full.shape}")
    return matches_full


def main():
    """
    Run the complete data processing pipeline.
    """
    print("Starting data processing pipeline...")
    
    print("\nStep 1: Merging FIFA rankings...")
    merged_rankings = merge()
    
    print("\nStep 2: Filtering matches (1993 onwards)...")
    filtered_df = filter_all_matches(min_year=1993)
    
    print("\nStep 3: Enriching matches with FIFA rankings...")
    enriched_df = enrich_dataset()
    
    print("\nPipeline complete!")
    print(f"  - Merged rankings: {len(merged_rankings)} rows")
    print(f"  - Filtered matches: {len(filtered_df)} rows")
    print(f"  - Enriched matches: {len(enriched_df)} rows")

def filter_relevant_teams(matches_path: Path = enriched_matches,
                          teams_path: Path = teams_master,
                          output_path: Path = relevant_matches,
                          require_both: bool = True,
                          drop_missing_rankings: bool = True) -> pd.DataFrame:
    """
    Keep only matches where teams are in teams_master.csv
    (qualified / still can qualify). Optionally require both teams
    and optionally drop rows with missing rankings.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(matches_path)
    teams = pd.read_csv(teams_path)

    # set of relevant teams
    teams["team"] = teams["team"].astype(str)
    relevant = set(teams["team"])

    # basic filter by team names
    if require_both:
        mask = matches["home_team"].isin(relevant) & matches["away_team"].isin(relevant)
    else:
        mask = matches["home_team"].isin(relevant) | matches["away_team"].isin(relevant)

    filtered = matches.loc[mask].copy()

    # optionally drop matches without rankings on either side
    if drop_missing_rankings:
        before = len(filtered)
        filtered = filtered.dropna(subset=["home_rank", "away_rank"])
        after = len(filtered)
        print(f"Dropped {before - after} matches with missing rankings")

    filtered.to_csv(output_path, index=False)
    print(f"Relevant matches saved to {output_path}")
    print(f"Shape: {filtered.shape}")
    return filtered



if __name__ == "__main__":
    main()