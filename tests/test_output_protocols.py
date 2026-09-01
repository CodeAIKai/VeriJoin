from __future__ import annotations

import pytest

from verijoin.output_protocols import parse_citation_output


def test_parse_citation_requires_integer_pair_references() -> None:
    answer, evidence = parse_citation_output(
        '<CITATION>{"answer":"Ada","evidence":[[0,1]]}</CITATION>'
    )
    assert answer == "Ada"
    assert evidence == {(0, 1)}
    with pytest.raises(TypeError):
        parse_citation_output(
            '<CITATION>{"answer":"Ada","evidence":[["0","1"]]}</CITATION>'
        )
    with pytest.raises(TypeError):
        parse_citation_output(
            '<CITATION>{"answer":"Ada","evidence":[[0]]}</CITATION>'
        )
