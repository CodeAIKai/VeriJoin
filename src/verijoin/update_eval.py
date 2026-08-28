from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .data import iter_examples
from .lineage import SourceField, bind_lineage, verify_lineage
from .maintenance import refresh_result
from .schema import Example
from .vm import execute, parse_program

CellKey = tuple[SourceField, int, int]


@dataclass(frozen=True, slots=True)
class UpdateSummary:
    dataset: str
    dataset_variant: str
    examples: int
    valid_programs: int
    certified_programs: int
    corpus_cells: int
    dependency_cells: int
    selective_recompute_rate: float
    recomputation_reduction_vs_whole_context: float
    mean_snapshot_bytes: float
    bind_p50_us: float
    bind_p95_us: float
    verify_p50_us: float
    verify_p95_us: float
    relevant_updates: int
    relevant_detected_rate: float
    relevant_replayed_rate: float
    relevant_recompile_rate: float
    irrelevant_updates: int
    irrelevant_preserved_rate: float
    irrelevant_reused_rate: float


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _corpus_cells(example: Example) -> set[CellKey]:
    cells: set[CellKey] = set()
    for document in example.documents:
        cells.add(("title", document.doc, -1))
        cells.update(("sentence", document.doc, sent) for sent in range(len(document.sentences)))
    return cells


def _mutate(example: Example, key: CellKey) -> Example:
    field, doc, sent = key
    if field == "question":
        return replace(example, question=example.question + " [updated]")
    documents = list(example.documents)
    document = documents[doc]
    if field == "title":
        documents[doc] = replace(document, title=document.title + " [updated]")
    else:
        sentences = list(document.sentences)
        sentences[sent] += " [updated]"
        documents[doc] = replace(document, sentences=tuple(sentences))
    return replace(example, documents=tuple(documents))


def evaluate_updates(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    limit: int | None = None,
    dataset_variant: str | None = None,
) -> UpdateSummary:
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    examples = valid = certified = corpus_total = dependency_total = snapshot_bytes = 0
    relevant = relevant_detected = relevant_replayed = relevant_recompile = 0
    irrelevant = irrelevant_preserved = irrelevant_reused = 0
    bind_times: list[float] = []
    verify_times: list[float] = []
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
        original = execute(example, program)
        if not original.valid:
            continue
        valid += 1
        started = time.perf_counter_ns()
        snapshot = bind_lineage(example, program)
        certified += int(snapshot.lineage_certified)
        bind_times.append((time.perf_counter_ns() - started) / 1_000.0)
        corpus = _corpus_cells(example)
        dependencies = {
            (cell.field, cell.doc, cell.sent)
            for cell in snapshot.cells
            if cell.field != "question"
        }
        corpus_total += len(corpus)
        dependency_total += len(dependencies)
        snapshot_bytes += len(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, separators=(",", ":")).encode()
        )

        if dependencies:
            relevant += 1
            changed = _mutate(example, min(dependencies))
            started = time.perf_counter_ns()
            check = verify_lineage(changed, snapshot)
            verify_times.append((time.perf_counter_ns() - started) / 1_000.0)
            relevant_detected += int(not check.current)
            refresh = refresh_result(changed, program, original.answer, snapshot)
            relevant_replayed += int(refresh.action == "replay")
            relevant_recompile += int(refresh.action == "recompile")
        untouched = corpus - dependencies
        if untouched:
            irrelevant += 1
            changed = _mutate(example, min(untouched))
            started = time.perf_counter_ns()
            check = verify_lineage(changed, snapshot)
            verify_times.append((time.perf_counter_ns() - started) / 1_000.0)
            irrelevant_preserved += int(check.current)
            refresh = refresh_result(changed, program, original.answer, snapshot)
            irrelevant_reused += int(refresh.action == "reuse")

    recompute = 100.0 * dependency_total / (corpus_total or 1)
    return UpdateSummary(
        dataset=dataset,
        dataset_variant=dataset_variant or ("distractor" if dataset == "hotpotqa" else "default"),
        examples=examples,
        valid_programs=valid,
        certified_programs=certified,
        corpus_cells=corpus_total,
        dependency_cells=dependency_total,
        selective_recompute_rate=recompute,
        recomputation_reduction_vs_whole_context=100.0 - recompute,
        mean_snapshot_bytes=snapshot_bytes / (valid or 1),
        bind_p50_us=_percentile(bind_times, 0.50),
        bind_p95_us=_percentile(bind_times, 0.95),
        verify_p50_us=_percentile(verify_times, 0.50),
        verify_p95_us=_percentile(verify_times, 0.95),
        relevant_updates=relevant,
        relevant_detected_rate=100.0 * relevant_detected / (relevant or 1),
        relevant_replayed_rate=100.0 * relevant_replayed / (relevant or 1),
        relevant_recompile_rate=100.0 * relevant_recompile / (relevant or 1),
        irrelevant_updates=irrelevant,
        irrelevant_preserved_rate=100.0 * irrelevant_preserved / (irrelevant or 1),
        irrelevant_reused_rate=100.0 * irrelevant_reused / (irrelevant or 1),
    )


def summary_dict(summary: UpdateSummary) -> dict[str, object]:
    return {
        **asdict(summary),
        "scope": (
            "synthetic one-cell updates; selective_recompute_rate is the uniform-cell "
            "expectation dependency_cells/corpus_cells, not an observed online event rate"
        ),
        "refresh_workload": (
            "relevant cells receive append-only changes, so stale plans can be replayed; "
            "counterfactual replacements and missing sources are reported by evaluate-attacks"
        ),
    }
