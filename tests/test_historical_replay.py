from __future__ import annotations

from verijoin.historical_replay import _present, _revision_content, _streams
from verijoin.text import normalize_answer


def _response(revisions: list[dict[str, object]]) -> dict[str, object]:
    return {"query": {"pages": [{"pageid": 1, "revisions": revisions}]}}


def test_streams_combines_deduplicates_and_sorts() -> None:
    old = {"revid": 1, "timestamp": "2017-10-01T00:00:00Z"}
    new = {"revid": 2, "timestamp": "2017-10-02T00:00:00Z"}
    payload = {
        "responses": [
            {
                "request_title": "Book X",
                "older": _response([old]),
                "newer": _response([old, new]),
            }
        ]
    }
    assert [row["revid"] for row in _streams(payload)["Book X"]] == [1, 2]


def test_revision_content_strips_common_mediawiki_markup() -> None:
    revision = {
        "slots": {
            "main": {
                "content": "Book X was written by [[Ada Writer|Ada]]. "
                "<ref>citation</ref>{{cite web|x=y}}"
            }
        }
    }
    content = normalize_answer(_revision_content(revision))
    assert _present(("Book X", "Ada"), content)
    assert "citation" not in content
