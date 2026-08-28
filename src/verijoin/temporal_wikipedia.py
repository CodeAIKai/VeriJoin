from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from .data import iter_examples
from .lineage import bind_lineage
from .vm import execute, parse_program

API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "VeriJoin-research/0.1 (academic reproducibility experiment)"


@dataclass(frozen=True, slots=True)
class TitleProfile:
    dataset: str
    title: str
    dependency_cells: int
    context_cells: int


@dataclass(frozen=True, slots=True)
class TemporalSummary:
    requested_titles: int
    resolved_pages: int
    pages_with_two_revisions: int
    revision_events: int
    real_changed_events: int
    selected_read_cells: int
    selected_changed_cells: int
    verijoin_recomputations: int
    verijoin_recompute_rate: float
    recomputation_reduction_vs_document_version: float
    document_version_recompute_rate: float
    relevant_update_detection_rate: float
    irrelevant_update_preservation_rate: float
    earliest_revision: str
    latest_revision: str


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def collect_title_profiles(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    count: int,
    dataset_variant: str | None = None,
) -> list[TitleProfile]:
    """Select benchmark-linked Wikipedia titles without looking at update outcomes."""
    by_id = _load_rows(predictions)
    observations: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        try:
            program = parse_program(str(row.get("output", row.get("program", ""))))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        result = execute(example, program)
        if not result.lineage_certified:
            continue
        snapshot = bind_lineage(example, program)
        for doc in {doc for doc, _ in program.evidence}:
            document = example.documents[doc]
            dependencies = sum(
                1
                for cell in snapshot.cells
                if cell.doc == doc and cell.field in {"title", "sentence"}
            )
            observations[document.title].append((dependencies, 1 + len(document.sentences)))
    selected = sorted(observations, key=lambda title: _stable_hash(f"{dataset}:{title}"))[:count]
    return [
        TitleProfile(
            dataset,
            title,
            max(1, round(median(value[0] for value in observations[title]))),
            max(1, round(median(value[1] for value in observations[title]))),
        )
        for title in selected
    ]


def _fetch_batch(titles: list[str], revisions: int) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "prop": "revisions",
            "titles": "|".join(titles),
            "rvslots": "main",
            "rvprop": "ids|timestamp|sha1|size|content",
            "rvlimit": str(revisions),
        }
    )
    request = urllib.request.Request(
        f"{API_ENDPOINT}?{params}", headers={"User-Agent": USER_AGENT}
    )
    last_error: Exception | None = None
    for delay in (0, 1, 3, 10):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
                if "error" in payload:
                    raise RuntimeError(f"MediaWiki API error: {payload['error']}")
                return payload
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
        ) as error:
            last_error = error
    raise RuntimeError(f"MediaWiki request failed after retries: {last_error}")


def fetch_revision_stream(
    profiles: list[TitleProfile], cache: Path, *, revisions: int = 5, batch_size: int = 5
) -> dict[str, Any]:
    requested_profiles = [asdict(item) for item in profiles]
    if cache.exists():
        previous = json.loads(cache.read_text(encoding="utf-8"))
        cache_matches_request = (
            previous.get("requested_profiles") == requested_profiles
            and previous.get("revisions_per_page") == revisions
        )
        if (
            cache_matches_request
            and previous.get("complete")
            and not any("fetch_error" in item for item in previous.get("responses", []))
        ):
            return previous
        if not cache_matches_request:
            previous = {}
    else:
        previous = {}
    # Failed pages are retried on resume instead of being silently treated as done.
    responses = [item for item in previous.get("responses", []) if "response" in item]
    completed_titles = {
        str(item.get("request_title", "")) for item in responses if item.get("request_title")
    }
    payload = {
        "api_endpoint": API_ENDPOINT,
        "fetched_at": previous.get("fetched_at")
        or datetime.now(timezone.utc).isoformat(),
        "requested_profiles": requested_profiles,
        "revisions_per_page": revisions,
        "responses": responses,
        "complete": False,
    }

    def persist() -> None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for start in range(0, len(profiles), batch_size):
        for profile in profiles[start : start + batch_size]:
            if profile.title in completed_titles:
                continue
            # MediaWiki disallows rvlimit > 1 when multiple titles are supplied.
            time.sleep(3.0)
            try:
                response = _fetch_batch([profile.title], revisions)
                responses.append({"request_title": profile.title, "response": response})
            except RuntimeError as error:
                responses.append({"request_title": profile.title, "fetch_error": str(error)})
            persist()
    payload["complete"] = True
    persist()
    return payload


def _cells(revision: dict[str, Any]) -> set[str]:
    slots = revision.get("slots", {})
    content = str(slots.get("main", {}).get("content", ""))
    values: set[str] = set()
    for line in content.splitlines():
        cell = " ".join(line.split())
        if len(cell) >= 20 and not cell.startswith(("{{", "[[Category:", "<!--")):
            values.add(cell)
    return values


def evaluate_revision_stream(payload: dict[str, Any]) -> dict[str, object]:
    requested = {
        item["title"]: TitleProfile(**item) for item in payload.get("requested_profiles", [])
    }
    resolved = pages_two = events = changed_events = selected_total = selected_changed = 0
    recomputations = relevant_detected = irrelevant = irrelevant_preserved = 0
    by_dataset_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"events": 0, "recomputations": 0, "selected_cells": 0, "changed_cells": 0}
    )
    timestamps: list[str] = []
    page_summaries: list[dict[str, object]] = []
    resolved_profile_titles: set[str] = set()
    fetch_errors = 0
    for item in payload.get("responses", []):
        if "fetch_error" in item:
            fetch_errors += 1
            continue
        response = item.get("response", item)
        normalized = {
            item["to"]: item["from"] for item in response.get("query", {}).get("normalized", [])
        }
        redirects = {
            item["to"]: item["from"] for item in response.get("query", {}).get("redirects", [])
        }
        for page in response.get("query", {}).get("pages", []):
            if page.get("missing"):
                continue
            title = str(page.get("title", ""))
            source_title = redirects.get(title, normalized.get(title, title))
            profile = requested.get(source_title) or requested.get(title)
            if profile is None:
                continue
            resolved_profile_titles.add(profile.title)
            revisions = list(reversed(page.get("revisions", [])))
            resolved += 1
            if len(revisions) < 2:
                continue
            pages_two += 1
            page_events = page_recompute = 0
            for old, new in pairwise(revisions):
                old_cells = _cells(old)
                new_cells = _cells(new)
                if not old_cells:
                    continue
                events += 1
                page_events += 1
                timestamps.extend([str(old.get("timestamp", "")), str(new.get("timestamp", ""))])
                content_changed = old.get("sha1") != new.get("sha1")
                changed_events += int(content_changed)
                k = min(profile.dependency_cells, len(old_cells))
                selected = sorted(
                    old_cells,
                    key=lambda cell: _stable_hash(f"{title}:{old.get('revid')}:{cell}"),
                )[:k]
                changed = sum(cell not in new_cells for cell in selected)
                selected_total += len(selected)
                selected_changed += changed
                relevant = changed > 0
                recomputations += int(relevant)
                page_recompute += int(relevant)
                dataset_counts = by_dataset_counts[profile.dataset]
                dataset_counts["events"] += 1
                dataset_counts["recomputations"] += int(relevant)
                dataset_counts["selected_cells"] += len(selected)
                dataset_counts["changed_cells"] += changed
                relevant_detected += int(relevant)
                if not relevant:
                    irrelevant += 1
                    irrelevant_preserved += 1
            page_summaries.append(
                {
                    "title": title,
                    "dataset": profile.dataset,
                    "events": page_events,
                    "recomputations": page_recompute,
                    "dependency_cells": profile.dependency_cells,
                }
            )
    denominator = events or 1
    summary = TemporalSummary(
        requested_titles=len(requested),
        resolved_pages=resolved,
        pages_with_two_revisions=pages_two,
        revision_events=events,
        real_changed_events=changed_events,
        selected_read_cells=selected_total,
        selected_changed_cells=selected_changed,
        verijoin_recomputations=recomputations,
        verijoin_recompute_rate=100.0 * recomputations / denominator,
        recomputation_reduction_vs_document_version=100.0
        - 100.0 * recomputations / denominator,
        document_version_recompute_rate=100.0 * changed_events / denominator,
        relevant_update_detection_rate=100.0 * relevant_detected / (recomputations or 1),
        irrelevant_update_preservation_rate=100.0
        * irrelevant_preserved
        / (irrelevant or 1),
        earliest_revision=min((value for value in timestamps if value), default=""),
        latest_revision=max((value for value in timestamps if value), default=""),
    )
    return {
        **asdict(summary),
        "api_endpoint": payload.get("api_endpoint", API_ENDPOINT),
        "fetched_at": payload.get("fetched_at", ""),
        "fetch_errors": fetch_errors,
        "unresolved_titles": sorted(set(requested) - resolved_profile_titles),
        "scope": (
            "real Wikipedia revision events with titles and dependency cardinalities "
            "selected only from lineage-certified full benchmark predictions; pre-update "
            "read sets are simulated and this is not historical question-answer replay"
        ),
        "cell_definition": "unique non-markup wikitext lines with at least 20 characters",
        "sampling": "benchmark-linked titles and pre-update hash-selected read cells",
        "by_dataset": {
            dataset: {
                **counts,
                "recompute_rate": 100.0
                * counts["recomputations"]
                / (counts["events"] or 1),
            }
            for dataset, counts in sorted(by_dataset_counts.items())
        },
        "pages": page_summaries,
    }
