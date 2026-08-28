import json

from verijoin import temporal_wikipedia
from verijoin.temporal_wikipedia import TitleProfile, evaluate_revision_stream


def _revision(revid: int, timestamp: str, sha1: str, lines: list[str]) -> dict[str, object]:
    return {
        "revid": revid,
        "timestamp": timestamp,
        "sha1": sha1,
        "slots": {"main": {"content": "\n".join(lines)}},
    }


def test_real_revision_metrics_preserve_unread_cells() -> None:
    old_cell = "This sufficiently long source cell exists before the real revision."
    new_cell = "This sufficiently long source cell exists after the real revision."
    payload = {
        "requested_profiles": [
            {
                "dataset": "hotpotqa",
                "title": "Page",
                "dependency_cells": 1,
                "context_cells": 2,
            }
        ],
        "responses": [
            {
                "query": {
                    "pages": [
                        {
                            "title": "Page",
                            "revisions": [
                                _revision(2, "2026-01-02T00:00:00Z", "new", [new_cell]),
                                _revision(1, "2026-01-01T00:00:00Z", "old", [old_cell]),
                            ],
                        }
                    ]
                }
            }
        ],
    }
    result = evaluate_revision_stream(payload)
    assert result["revision_events"] == 1
    assert result["real_changed_events"] == 1
    assert result["document_version_recompute_rate"] == 100.0
    assert result["relevant_update_detection_rate"] == 100.0


def test_revision_cache_is_reused_only_for_the_same_request(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "revisions.json"
    cache.write_text(
        json.dumps(
            {
                "complete": True,
                "revisions_per_page": 5,
                "requested_profiles": [
                    {
                        "dataset": "hotpotqa",
                        "title": "Old page",
                        "dependency_cells": 1,
                        "context_cells": 2,
                    }
                ],
                "responses": [],
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[list[str], int]] = []

    def fake_fetch(titles: list[str], revisions: int) -> dict[str, object]:
        calls.append((titles, revisions))
        return {"query": {"pages": []}}

    monkeypatch.setattr(temporal_wikipedia, "_fetch_batch", fake_fetch)
    monkeypatch.setattr(temporal_wikipedia.time, "sleep", lambda _: None)
    profile = TitleProfile("2wiki", "New page", 2, 3)
    result = temporal_wikipedia.fetch_revision_stream([profile], cache, revisions=4)

    assert calls == [(["New page"], 4)]
    assert result["requested_profiles"] == [
        {
            "dataset": "2wiki",
            "title": "New page",
            "dependency_cells": 2,
            "context_cells": 3,
        }
    ]
    assert result["revisions_per_page"] == 4
