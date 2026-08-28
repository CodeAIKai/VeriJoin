from collections import Counter

import pytest

from verijoin.data import assert_public_record, parse_hotpot
from verijoin.sft import _balanced_subset


def test_public_record_accepts_documents() -> None:
    assert_public_record(
        {"qid": "x", "question": "q", "documents": [{"title": "t", "sentences": ["s"]}]}
    )


@pytest.mark.parametrize(
    "record",
    [
        {"answer": "leak"},
        {"nested": {"supporting_facts": []}},
        {"documents": [{"is_supporting": True}]},
        {"gold_program": {}},
    ],
)
def test_public_record_rejects_supervision(record: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        assert_public_record(record)


def test_hotpot_marks_missing_fullwiki_gold_as_incomplete() -> None:
    row = {
        "id": "x",
        "question": "Where?",
        "answer": "Delta",
        "type": "bridge",
        "context": {
            "title": ["Present"],
            "sentences": [["Present links to Missing."]],
        },
        "supporting_facts": {
            "title": ["Present", "Missing"],
            "sent_id": [0, 0],
        },
    }
    example = parse_hotpot(row, "dev")
    assert example.support == ((0, 0),)
    assert example.support_expected == 2
    assert not example.support_complete


def test_hotpot_unescapes_support_titles_before_mapping() -> None:
    row = {
        "id": "html",
        "question": "Who?",
        "answer": "Ada",
        "type": "bridge",
        "context": {
            "title": ["Tunnels & Trolls"],
            "sentences": [["Ada designed it."]],
        },
        "supporting_facts": {"title": ["Tunnels &amp; Trolls"], "sent_id": [0]},
    }
    example = parse_hotpot(row, "dev")
    assert example.support == ((0, 0),)
    assert example.support_complete


def test_hotpot_preserves_empty_sentence_index_placeholders() -> None:
    row = {
        "id": "empty",
        "question": "Who?",
        "answer": "Ada",
        "type": "bridge",
        "context": {
            "title": ["Document"],
            "sentences": [["", "Ada wrote it."]],
        },
        "supporting_facts": {"title": ["Document"], "sent_id": [1]},
    }
    example = parse_hotpot(row, "dev")
    assert example.documents[0].sentences == ("", "Ada wrote it.")
    assert example.support == ((0, 1),)


def test_validation_subset_balances_datasets_before_operators() -> None:
    rows = [
        {
            "id": f"{dataset}-{operator}-{index}",
            "dataset": dataset,
            "program": {"answer": {"op": operator}},
        }
        for dataset, operators in {"hotpotqa": ("copy",), "2wiki": ("copy", "equal")}.items()
        for operator in operators
        for index in range(10)
    ]
    subset = _balanced_subset(rows, 8, 7)
    counts = Counter(row["dataset"] for row in subset)
    operators = Counter((row["dataset"], row["program"]["answer"]["op"]) for row in subset)
    assert counts == {"2wiki": 4, "hotpotqa": 4}
    assert operators[("2wiki", "copy")] == operators[("2wiki", "equal")] == 2
