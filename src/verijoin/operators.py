from __future__ import annotations

import re
from typing import Literal

from .text import normalize_answer

Reducer = Literal["argmin", "argmax", "equal", "common"]

_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        1,
    )
}

_DEMonyms = {
    "american": "united states",
    "british": "united kingdom",
    "english": "united kingdom",
    "scottish": "united kingdom",
    "welsh": "united kingdom",
    "french": "france",
    "german": "germany",
    "italian": "italy",
    "spanish": "spain",
    "canadian": "canada",
    "australian": "australia",
    "indian": "india",
    "japanese": "japan",
    "chinese": "china",
    "russian": "russia",
    "croatian": "croatia",
    "czech": "czech republic",
    "ugandan": "uganda",
    "malaysian": "malaysia",
}


def canonical_scalar(value: str) -> str:
    normalized = normalize_answer(value)
    return _DEMonyms.get(normalized, normalized)


def ordered_scalar(value: str) -> tuple[float, ...] | None:
    """Parse an exact quoted date/year/number for deterministic ordering."""
    lowered = value.casefold().replace("–", "-").replace("—", "-")
    years = re.findall(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)", lowered)
    if years:
        year = int(years[0])
        month = next((number for name, number in _MONTHS.items() if name in lowered), 0)
        day = 0
        if month:
            before = re.search(r"(?<!\d)([0-3]?\d)\s+(?:" + "|".join(_MONTHS) + r")", lowered)
            after = re.search(r"(?:" + "|".join(_MONTHS) + r")\s+([0-3]?\d)", lowered)
            match = before or after
            day = int(match.group(1)) if match else 0
        return float(year), float(month), float(day)
    compact = re.sub(r"(?<=\d),(?=\d)", "", lowered)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", compact)
    return (float(match.group()),) if match else None


def reducer_from_question(question: str) -> Literal["argmin", "argmax"] | None:
    lowered = question.casefold()
    maximum = (
        "younger",
        "youngest",
        "later",
        "latest",
        "newer",
        "newest",
        "more ",
        "most ",
        "larger",
        "largest",
        "longer",
        "longest",
        "higher",
        "highest",
        "greater",
    )
    minimum = (
        "older",
        "oldest",
        "earlier",
        "earliest",
        "first",
        "less ",
        "fewer",
        "smaller",
        "smallest",
        "shorter",
        "shortest",
        "lower",
        "lowest",
    )
    if any(cue in lowered for cue in maximum):
        return "argmax"
    if any(cue in lowered for cue in minimum):
        return "argmin"
    return None
