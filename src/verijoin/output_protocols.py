from __future__ import annotations

import json
import re

_CITATION = re.compile(r"<CITATION>\s*(\{.*?\})\s*</CITATION>", flags=re.DOTALL)


def parse_citation_output(output: str) -> tuple[str, set[tuple[int, int]]]:
    """Parse the citation contract without coercing non-integer references."""
    match = _CITATION.search(output)
    if match is None:
        raise ValueError("missing CITATION wrapper")
    payload = json.loads(match.group(1))
    answer = payload["answer"]
    raw_evidence = payload["evidence"]
    if not isinstance(answer, str) or not isinstance(raw_evidence, list):
        raise TypeError("citation answer must be a string and evidence must be a list")
    evidence: set[tuple[int, int]] = set()
    for reference in raw_evidence:
        if (
            not isinstance(reference, list)
            or len(reference) != 2
            or type(reference[0]) is not int
            or type(reference[1]) is not int
        ):
            raise TypeError("each citation reference must be two JSON integers")
        evidence.add((reference[0], reference[1]))
    return answer, evidence
