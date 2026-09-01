from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from .data import iter_examples
from .schema import Program, SpanPointer
from .text import normalize_answer
from .vm import execute, parse_program


def _revision_content(revision: dict[str, Any]) -> str:
    raw = str(revision.get("slots", {}).get("main", {}).get("content", ""))
    value = html.unescape(raw)
    value = re.sub(r"<!--.*?-->", " ", value, flags=re.DOTALL)
    value = re.sub(r"<ref\b[^>]*>.*?</ref>", " ", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\{\{[^{}]*\}\}", " ", value)
    value = re.sub(r"'{2,}", "", value)
    return " ".join(value.split())


def _page_revisions(response: dict[str, Any]) -> list[dict[str, Any]]:
    pages = response.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return []
    return list(pages[0].get("revisions", []))


def _streams(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    streams: dict[str, list[dict[str, Any]]] = {}
    for item in payload.get("responses", []):
        if "fetch_error" in item:
            continue
        revisions = _page_revisions(item["older"]) + _page_revisions(item["newer"])
        unique = {int(revision["revid"]): revision for revision in revisions}
        streams[str(item["request_title"])] = sorted(
            unique.values(), key=lambda revision: str(revision.get("timestamp", ""))
        )
    return streams


def _pointer_key(pointer: SpanPointer) -> tuple[str, int, int, str]:
    return pointer.field, pointer.doc, pointer.sent, pointer.quote


def _program_quotes(program: Program, doc: int) -> tuple[str, ...]:
    answer = program.answer
    pointers = ([answer.pointer] if answer.pointer is not None else [])
    pointers += list(answer.operands) + list(answer.labels)
    pointers += [join.left for join in program.joins]
    values = {
        pointer.quote
        for pointer in pointers
        if pointer.doc == doc and pointer.quote
    }
    values.update(
        join.left.quote
        for join in program.joins
        if join.right_doc == doc and join.left.quote
    )
    return tuple(sorted(values))


def _present(values: tuple[str, ...], normalized_content: str) -> bool:
    return all(normalize_answer(value) in normalized_content for value in values)


def evaluate_hotpot_history(
    raw_root: Path,
    predictions: Path,
    history: Path,
) -> dict[str, object]:
    payload = json.loads(history.read_text(encoding="utf-8"))
    streams = _streams(payload)
    rows = {
        str(row["id"]): row
        for row in (
            json.loads(line)
            for line in predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    linked: dict[str, list[tuple[str, tuple[str, ...], tuple[str, ...]]]] = defaultdict(list)
    certified = linked_program_docs = 0
    for example in iter_examples("hotpotqa", raw_root, "dev", "distractor"):
        row = rows.get(example.qid)
        if row is None:
            continue
        try:
            program = parse_program(str(row.get("output", "")))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if not execute(example, program).lineage_certified:
            continue
        certified += 1
        for doc in sorted({doc for doc, _ in program.evidence}):
            title = example.documents[doc].title
            if title not in streams:
                continue
            sentences = tuple(
                example.documents[doc].sentences[sent]
                for source_doc, sent in program.evidence
                if source_doc == doc
            )
            quotes = _program_quotes(program, doc)
            if not sentences or not quotes:
                continue
            linked[title].append((example.qid, sentences, quotes))
            linked_program_docs += 1

    events = changed_events = reuse = replay = recompile = 0
    initially_bound_program_docs: set[tuple[str, str]] = set()
    event_rows: list[dict[str, object]] = []
    for title, programs in linked.items():
        revisions = streams[title]
        for old, new in pairwise(revisions):
            old_content = normalize_answer(_revision_content(old))
            new_content = normalize_answer(_revision_content(new))
            page_changed = old.get("sha1") != new.get("sha1")
            for qid, sentences, quotes in programs:
                if not _present(sentences, old_content) or not _present(quotes, old_content):
                    continue
                initially_bound_program_docs.add((qid, title))
                events += 1
                changed_events += int(page_changed)
                if not page_changed or _present(sentences, new_content):
                    action = "reuse"
                    reuse += 1
                elif _present(quotes, new_content):
                    action = "replay"
                    replay += 1
                else:
                    action = "recompile"
                    recompile += 1
                event_rows.append(
                    {
                        "qid": qid,
                        "title": title,
                        "old_revid": old.get("revid"),
                        "new_revid": new.get("revid"),
                        "old_timestamp": old.get("timestamp"),
                        "new_timestamp": new.get("timestamp"),
                        "page_changed": page_changed,
                        "action": action,
                        "read_sentences": len(sentences),
                        "pointer_quotes": len(quotes),
                    }
                )
    denominator = events or 1
    changed_denominator = changed_events or 1
    return {
        "dataset": "hotpotqa",
        "official_snapshot_anchor": payload.get("anchor", ""),
        "requested_titles": len(payload.get("requested_titles", [])),
        "resolved_titles": len(streams),
        "certified_programs": certified,
        "linked_program_documents": linked_program_docs,
        "initially_bound_program_documents": len(initially_bound_program_docs),
        "program_revision_events": events,
        "changed_program_revision_events": changed_events,
        "reuse": reuse,
        "replay": replay,
        "recompile": recompile,
        "reuse_rate": 100.0 * reuse / denominator,
        "replay_rate": 100.0 * replay / denominator,
        "recompile_rate": 100.0 * recompile / denominator,
        "llm_call_reduction_vs_document_invalidation": 100.0
        * (changed_events - recompile)
        / changed_denominator,
        "event_rows": event_rows,
        "scope": (
            "real MediaWiki revisions anchored at HotpotQA's official 2017-10-01 dump; "
            "only events whose actual VeriJoin evidence sentences and pointer quotes bind "
            "to the old revision are eligible"
        ),
        "normalization": (
            "MediaWiki markup is stripped and both revision text and benchmark cells use "
            "the benchmark answer normalizer before exact substring binding"
        ),
        "action_semantics": (
            "reuse means every cited sentence persists; replay means cited wording changed "
            "but every program quote persists; recompile means at least one required quote vanished"
        ),
    }
