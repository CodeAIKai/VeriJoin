from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .data import iter_examples
from .lineage import SourceField, bind_lineage, verify_lineage
from .schema import Example, Program
from .vm import execute, parse_program

CellKey = tuple[SourceField, int, int]


@dataclass(frozen=True, slots=True)
class ProvenanceComparison:
    dataset: str
    examples: int
    valid_programs: int
    certified_programs: int
    verijoin_mean_cells: float
    verijoin_mean_bytes: float
    verijoin_bind_p50_us: float
    verijoin_verify_p50_us: float
    blip_style_mean_cells: float
    blip_style_mean_bytes: float
    blip_style_replay_success_rate: float
    blip_style_extra_calls_per_example: float
    blip_style_p50_ms: float
    groundedcache_relevant_update_unsafe_hit_rate: float
    groundedcache_same_document_false_invalidation_rate: float
    groundedcache_version_check_p50_us: float
    verijoin_same_document_false_invalidation_rate: float


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _corpus_cells(example: Example) -> tuple[CellKey, ...]:
    values: list[CellKey] = []
    for document in example.documents:
        values.append(("title", document.doc, -1))
        values.extend(("sentence", document.doc, sent) for sent in range(len(document.sentences)))
    return tuple(values)


def _source(example: Example, key: CellKey) -> str:
    field, doc, sent = key
    if field == "question":
        return example.question
    document = example.documents[doc]
    return document.title if field == "title" else document.sentences[sent]


def _blank(example: Example, key: CellKey) -> Example:
    field, doc, sent = key
    if field == "question":
        return replace(example, question="")
    documents = list(example.documents)
    document = documents[doc]
    if field == "title":
        documents[doc] = replace(document, title="")
    else:
        sentences = list(document.sentences)
        sentences[sent] = ""
        documents[doc] = replace(document, sentences=tuple(sentences))
    return replace(example, documents=tuple(documents))


def _mutate(example: Example, key: CellKey) -> Example:
    value = _source(example, key) + " [real-version-change]"
    field, doc, sent = key
    if field == "question":
        return replace(example, question=value)
    documents = list(example.documents)
    document = documents[doc]
    if field == "title":
        documents[doc] = replace(document, title=value)
    else:
        sentences = list(document.sentences)
        sentences[sent] = value
        documents[doc] = replace(document, sentences=tuple(sentences))
    return replace(example, documents=tuple(documents))


def _document_digest(example: Example, doc: int) -> str:
    document = example.documents[doc]
    value = json.dumps([document.title, *document.sentences], ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def _blip_style_deletion(
    example: Example, program: Program, answer: str
) -> tuple[set[CellKey], bool, int, int, float]:
    """A one-pass proxy for black-box deletion provenance.

    Each VM replay is substantially cheaper than an LLM call, so measured replay
    latency is a favorable cost lower bound for this BLIP-style baseline. The
    provenance itself is not claimed to be a lower bound or to reproduce the
    unavailable official BLIP implementation.
    """
    started = time.perf_counter_ns()
    working = example
    kept = set(_corpus_cells(example))
    calls = 0
    for key in _corpus_cells(example):
        trial = _blank(working, key)
        result = execute(trial, program)
        calls += 1
        if result.valid and result.answer == answer:
            working = trial
            kept.remove(key)
    replay = execute(working, program)
    calls += 1
    payload = [(key, _source(example, key)) for key in sorted(kept)]
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return kept, replay.valid and replay.answer == answer, calls, size, elapsed_ms


def compare_provenance_baselines(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    limit: int | None = 1000,
    dataset_variant: str | None = None,
) -> dict[str, object]:
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    examples = valid = certified = 0
    v_cells = v_bytes = b_cells = b_bytes = b_calls = b_replay = 0
    bind_us: list[float] = []
    verify_us: list[float] = []
    blip_ms: list[float] = []
    relevant = grounded_unsafe = same_doc = grounded_false = verijoin_false = 0
    grounded_us: list[float] = []
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        if limit is not None and examples >= limit:
            break
        examples += 1
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        try:
            program = parse_program(str(row.get("output", row.get("program", ""))))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        result = execute(example, program)
        if not result.valid:
            continue
        valid += 1
        started = time.perf_counter_ns()
        snapshot = bind_lineage(example, program)
        if not snapshot.lineage_certified:
            continue
        certified += 1
        bind_us.append((time.perf_counter_ns() - started) / 1_000.0)
        dependencies = {(cell.field, cell.doc, cell.sent) for cell in snapshot.cells}
        v_cells += len(dependencies)
        v_bytes += len(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, separators=(",", ":")).encode()
        )

        kept, replayed, calls, size, elapsed_ms = _blip_style_deletion(
            example, program, result.answer
        )
        b_cells += len(kept)
        b_bytes += size
        b_calls += calls
        b_replay += int(replayed)
        blip_ms.append(elapsed_ms)

        corpus = set(_corpus_cells(example))
        corpus_dependencies = {key for key in dependencies if key[0] != "question"}
        if corpus_dependencies:
            relevant += 1
            changed = _mutate(example, min(corpus_dependencies))
            started = time.perf_counter_ns()
            current = verify_lineage(changed, snapshot).current
            verify_us.append((time.perf_counter_ns() - started) / 1_000.0)
            cited_docs = {doc for doc, _ in program.evidence}
            old_versions = {doc: _document_digest(example, doc) for doc in cited_docs}
            started = time.perf_counter_ns()
            grounded_current = all(
                _document_digest(changed, doc) == version
                for doc, version in old_versions.items()
            )
            grounded_us.append((time.perf_counter_ns() - started) / 1_000.0)
            grounded_unsafe += int(grounded_current)
            if current:
                raise AssertionError("dependency mutation escaped VeriJoin lineage")

        evidence_docs = {doc for doc, _ in program.evidence}
        unread_same_doc = sorted(
            key for key in corpus - corpus_dependencies if key[1] in evidence_docs
        )
        if unread_same_doc:
            same_doc += 1
            changed = _mutate(example, unread_same_doc[0])
            started = time.perf_counter_ns()
            v_current = verify_lineage(changed, snapshot).current
            verify_us.append((time.perf_counter_ns() - started) / 1_000.0)
            old_versions = {doc: _document_digest(example, doc) for doc in evidence_docs}
            started = time.perf_counter_ns()
            grounded_current = all(
                _document_digest(changed, doc) == version
                for doc, version in old_versions.items()
            )
            grounded_us.append((time.perf_counter_ns() - started) / 1_000.0)
            grounded_false += int(not grounded_current)
            verijoin_false += int(not v_current)

    denominator = certified or 1
    comparison = ProvenanceComparison(
        dataset=dataset,
        examples=examples,
        valid_programs=valid,
        certified_programs=certified,
        verijoin_mean_cells=v_cells / denominator,
        verijoin_mean_bytes=v_bytes / denominator,
        verijoin_bind_p50_us=_percentile(bind_us, 0.5),
        verijoin_verify_p50_us=_percentile(verify_us, 0.5),
        blip_style_mean_cells=b_cells / denominator,
        blip_style_mean_bytes=b_bytes / denominator,
        blip_style_replay_success_rate=100.0 * b_replay / denominator,
        blip_style_extra_calls_per_example=b_calls / denominator,
        blip_style_p50_ms=_percentile(blip_ms, 0.5),
        groundedcache_relevant_update_unsafe_hit_rate=100.0
        * grounded_unsafe
        / (relevant or 1),
        groundedcache_same_document_false_invalidation_rate=100.0
        * grounded_false
        / (same_doc or 1),
        groundedcache_version_check_p50_us=_percentile(grounded_us, 0.5),
        verijoin_same_document_false_invalidation_rate=100.0
        * verijoin_false
        / (same_doc or 1),
    )
    return {
        **asdict(comparison),
        "baseline_scope": {
            "eligibility": "lineage-certified program outputs only",
            "blip": (
                "one-pass deletion proxy with deterministic VM replay; replay cost is a "
                "favorable lower bound, provenance is not; not the official implementation"
            ),
            "groundedcache": (
                "official exact-repeat gate semantics with each benchmark document as a chunk; "
                "query/evidence/support gates pass and source-version gate decides"
            ),
        },
    }
