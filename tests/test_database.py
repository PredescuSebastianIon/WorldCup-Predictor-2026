"""
Unit tests for the database helper functions.

These tests use an in‑memory SQLite database instead of the on‑disk
``winner_predictions.db`` used by the application.  This allows the
functions to be exercised without touching the filesystem and ensures
tests remain isolated and repeatable.
"""

import sqlite3

from pathlib import Path

# Ensure ../data relative to project root exists, since database.py uses "../data/..."
ROOT = Path(__file__).resolve().parents[1]     # WorldCup-Predictor-2026
external_data_dir = ROOT.parent / "data"       # ~/data
external_data_dir.mkdir(parents=True, exist_ok=True)

import src.database as db



def _create_memory_db():
    """Create an in‑memory SQLite database with the expected table."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE winner(user VARCHAR(50), team VARCHAR(50));"
    )
    return conn, cursor


def test_update_procentage_computes_percentages():
    """``update_procentage`` should return correct counts and percentages."""
    # Use in-memory DB
    conn, cursor = _create_memory_db()
    # Insert sample data: two votes for A and one for B
    cursor.executemany(
        "INSERT INTO winner (user, team) VALUES (?, ?)",
        [("u1", "A"), ("u2", "A"), ("u3", "B")],
    )
    conn.commit()
    result = db.update_procentage(cursor)
    # Should have two rows: A and B
    assert set(result["team"]) == {"A", "B"}
    # Check counts
    cnts = dict(zip(result["team"], result["cnt"]))
    assert cnts["A"] == 2
    assert cnts["B"] == 1
    # Check percentages (floating point tolerance)
    pcts = dict(zip(result["team"], result["percentage"]))
    assert abs(pcts["A"] - 2 / 3) < 1e-6
    assert abs(pcts["B"] - 1 / 3) < 1e-6


def test_select_winner_inserts_and_returns_updated():
    """``select_winner`` should insert a new row and return updated percentages."""
    conn, cursor = _create_memory_db()
    # Pre-populate one entry for team A
    cursor.execute(
        "INSERT INTO winner (user, team) VALUES ('u1','A')"
    )
    conn.commit()
    # Insert a new vote for team B
    result = db.select_winner(
        user="u2",
        team="B",
        winner_predictions_cursor=cursor,
        winner_predictions_db=conn,
    )
    # After insertion there should be entries for both teams
    assert set(result["team"]) == {"A", "B"}
    # Both should have 50% of the votes
    assert all(abs(result["percentage"] - 0.5) < 1e-6)