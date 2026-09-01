from __future__ import annotations

from verijoin.recompile_recovery import mutate_numeric_quote


def test_mutate_numeric_quote_changes_one_year() -> None:
    assert mutate_numeric_quote("12 June 1516") == "12 June 1517"


def test_mutate_numeric_quote_preserves_commas_and_decimals() -> None:
    assert mutate_numeric_quote("2,100") == "2,101"
    assert mutate_numeric_quote("3.50") == "4.50"
    assert mutate_numeric_quote("21st century") == "22nd century"


def test_mutate_numeric_quote_rejects_ambiguous_range() -> None:
    assert mutate_numeric_quote("1993-94") is None
