"""
Tests for the data processing pipeline functions.

These tests exercise the core functionality in src/data_process.  The
functions work with CSV files on disk, so the tests make use of tmp_path
to create temporary input and output files.  Small in-memory data frames
representing matches and rankings are written to disk and then passed to
filter_all_matches, enrich_dataset, filter_relevant_teams and
split_dataset.  The resulting data frames are read back into memory
and various properties are asserted to ensure that the logic behaves as
expected.

Using minimal synthetic data avoids any dependency on the large, real data
files in the repository while still covering the important code paths.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import src.data_process as dp


def test_filter_all_matches_min_year(tmp_path):
    """Matches before ``min_year`` should be filtered out."""
    # Build a small matches dataset covering multiple years
    data = {
        "date": ["1990-07-01", "1995-05-10", "2000-03-15", "2020-11-20"],
        "home_team": ["A", "B", "C", "D"],
        "away_team": ["X", "Y", "Z", "W"],
        "home_score": [1, 0, 2, 3],
        "away_score": [0, 1, 2, 0],
    }
    df = pd.DataFrame(data)
    input_csv = tmp_path / "all_matches.csv"
    df.to_csv(input_csv, index=False)

    output_csv = tmp_path / "filtered.csv"
    # Filter from 1995 onwards
    filtered = dp.filter_all_matches(
        input_path=Path(input_csv), output_path=Path(output_csv), min_year=1995
    )
    # Expect three rows (1995, 2000 and 2020)
    assert len(filtered) == 3
    assert filtered["date"].min().year >= 1995
    assert list(filtered["date"].dt.year) == [1995, 2000, 2020]


def test_enrich_dataset_assigns_latest_rank(tmp_path):
    """Ensure enrich_dataset merges rankings based on the latest snapshot before each match."""
    # Create a simple matches CSV
    matches = pd.DataFrame(
        {
            "date": ["2020-01-05", "2020-06-01"],
            "home_team": ["TeamA", "TeamB"],
            "away_team": ["TeamB", "TeamA"],
            "home_score": [1, 2],
            "away_score": [0, 1],
        }
    )
    matches_path = tmp_path / "matches.csv"
    matches.to_csv(matches_path, index=False)

    # Create a rankings CSV with two snapshots per team
    rankings = pd.DataFrame(
        {
            "date": [
                "2019-12-31",
                "2020-05-20",
                "2019-12-31",
                "2020-05-20",
            ],
            "country": ["TeamA", "TeamA", "TeamB", "TeamB"],
            "abbreviation": ["A", "A", "B", "B"],
            "rank": [10, 9, 5, 4],
            "prev_rank": [12, 10, 7, 5],
            "points": [1000, 1100, 1500, 1600],
            "prev_points": [900, 1000, 1400, 1500],
            "confed": ["UEFA", "UEFA", "UEFA", "UEFA"],
        }
    )
    rankings_path = tmp_path / "ranks.csv"
    rankings.to_csv(rankings_path, index=False)

    output_csv = tmp_path / "enriched.csv"
    enriched = dp.enrich_dataset(
        matches_path=Path(matches_path),
        rankings_path=Path(rankings_path),
        output_path=Path(output_csv),
    )

    # For the first match (TeamA vs TeamB on 2020‑01‑05), the latest snapshot before
    # 2020‑01‑05 is 2019‑12‑31
    row0 = enriched.loc[0]
    assert row0["home_rank"] == 10
    assert row0["away_rank"] == 5

    # For the second match (TeamB vs TeamA on 2020‑06‑01), the latest snapshot
    # before 2020‑06‑01 is 2020‑05‑20
    row1 = enriched.loc[1]
    assert row1["home_rank"] == 4
    assert row1["away_rank"] == 9

    # Ensure derived features are computed
    assert "rank_diff" in enriched.columns
    assert "points_diff" in enriched.columns
    # All teams share the same confederation in this synthetic data
    assert (enriched["same_confed"] == 1).all()


def test_filter_relevant_teams(tmp_path):
    """Test filtering by relevant teams and dropping rows with missing rankings."""
    # Build enriched matches CSV with ranking columns
    matches = pd.DataFrame(
        {
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["B", "A", "E", "D"],
            "home_rank": [1, 2, np.nan, 4],
            "away_rank": [2, 1, 3, np.nan],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
        }
    )
    matches_path = tmp_path / "enriched.csv"
    matches.to_csv(matches_path, index=False)

    # teams_master lists teams A, B and D only
    teams = pd.DataFrame({"team": ["A", "B", "D"]})
    teams_path = tmp_path / "teams_master.csv"
    teams.to_csv(teams_path, index=False)

    out_path = tmp_path / "relevant.csv"
    # require both teams and drop missing rankings
    filtered = dp.filter_relevant_teams(
        matches_path=Path(matches_path),
        teams_path=Path(teams_path),
        output_path=Path(out_path),
        require_both=True,
        drop_missing_rankings=True,
    )
    # Only first two rows should remain (A vs B and B vs A) and both have full ranks
    assert len(filtered) == 2
    assert set(filtered["home_team"]) == {"A", "B"}
    assert set(filtered["away_team"]) == {"B", "A"}

    # When require_both=False, rows with at least one relevant team remain
    filtered2 = dp.filter_relevant_teams(
        matches_path=Path(matches_path),
        teams_path=Path(teams_path),
        output_path=tmp_path / "relevant2.csv",
        require_both=False,
        drop_missing_rankings=True,
    )
    # Row with C vs E is dropped because neither team is relevant, row with D vs D is
    # dropped because of missing ranks; first two rows remain
    assert len(filtered2) == 2


def test_split_dataset(tmp_path):
    """Validate that ``split_dataset`` performs a time‑based split correctly."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2017-12-31",
                    "2018-01-01",
                    "2019-06-01",
                    "2022-01-01",
                    "2023-05-05",
                ]
            ),
            "home_team": ["A"] * 5,
            "away_team": ["B"] * 5,
        }
    )
    input_path = tmp_path / "relevant.csv"
    df.to_csv(input_path, index=False)

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"

    train_df, val_df, test_df = dp.split_dataset(
        input_path=Path(input_path),
        train_path=Path(train_path),
        val_path=Path(val_path),
        test_path=Path(test_path),
        val_start="2018-01-01",
        test_start="2022-01-01",
    )

    # verify counts
    assert len(train_df) == 1  # only 2017-12-31
    assert len(val_df) == 3  # 2018 and 2019 dates
    assert len(test_df) == 1  # 2022 and beyond