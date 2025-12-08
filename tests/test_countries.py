"""
Simple sanity checks for the countries module.

The module defines a list of nations participating in the
upcoming tournament.  Tests verify that there are no duplicates and
that a few expected teams are present.
"""

import src.countries as countries


def test_countries_unique_and_length():
    """The list of countries should contain 68 unique entries."""
    assert len(countries.countries) == 68
    # ensure all entries are unique
    assert len(set(countries.countries)) == len(countries.countries)


def test_countries_contains_known_teams():
    """Check that some well known teams are present in the list."""
    expected = {"France", "Brazil", "Argentina", "Germany"}
    assert expected.issubset(set(countries.countries))