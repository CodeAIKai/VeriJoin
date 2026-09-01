from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .data import iter_examples
from .lineage import SourceCell, bind_lineage
from .maintenance import refresh_result
from .schema import Example, Program, SpanPointer
from .vm import execute, parse_program


@dataclass(slots=True)
class ActionCounts:
    eligible: int = 0
    reuse: int = 0
    replay: int = 0
    recompile: int = 0
    expected_action: str = ""
    expected_action_hits: int = 0
    oracle_repair_valid: int = 0
    oracle_answer_changed: int = 0

    def record(self, action: str) -> None:
        self.eligible += 1
        setattr(self, action, getattr(self, action) + 1)
        self.expected_action_hits += int(action == self.expected_action)

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "expected_action_rate": 100.0
            * self.expected_action_hits
            / (self.eligible or 1),
            "oracle_repair_valid_rate": 100.0
            * self.oracle_repair_valid
            / (self.eligible or 1),
            "oracle_answer_change_rate": 100.0
            * self.oracle_answer_changed
            / (self.oracle_repair_valid or 1),
        }


def _source(example: Example, pointer: SpanPointer) -> str:
    if pointer.field == "question":
        return example.question
    document = example.documents[pointer.doc]
    if pointer.field == "title":
        return document.title
    return document.sentences[pointer.sent]


def _replace_source(example: Example, pointer: SpanPointer, old: str, new: str) -> Example:
    if old not in _source(example, pointer):
        raise ValueError("replacement target is absent")
    if pointer.field == "question":
        return replace(example, question=example.question.replace(old, new))
    documents = list(example.documents)
    document = documents[pointer.doc]
    if pointer.field == "title":
        documents[pointer.doc] = replace(document, title=document.title.replace(old, new))
    else:
        sentences = list(document.sentences)
        sentences[pointer.sent] = sentences[pointer.sent].replace(old, new)
        documents[pointer.doc] = replace(document, sentences=tuple(sentences))
    return replace(example, documents=tuple(documents))


def _append_to_cell(example: Example, cell: SourceCell, suffix: str) -> Example:
    pointer = SpanPointer(cell.doc, cell.sent, -1, -1, cell.field, "")
    source = _source(example, pointer)
    return _replace_source(example, pointer, source, source + suffix)


def _cell_key(cell: SourceCell) -> tuple[str, int, int]:
    return cell.field, cell.doc, cell.sent


def _all_context_cells(example: Example) -> list[SourceCell]:
    cells: list[SourceCell] = []
    for document in example.documents:
        cells.append(SourceCell("title", document.doc, -1, ""))
        cells.extend(
            SourceCell("sentence", document.doc, sent, "")
            for sent in range(len(document.sentences))
        )
    return cells


def _replacement(quote: str) -> str:
    compact = quote.strip().replace(",", "")
    candidates: list[str] = []
    if re.fullmatch(r"-?\d+", compact):
        candidates.append(str(int(compact) + 17))
    if re.fullmatch(r"-?\d+\.\d+", compact):
        candidates.append(str(float(compact) + 17.0))
    normalized = quote.strip().lower()
    if normalized == "yes":
        candidates.append("no")
    if normalized == "no":
        candidates.append("yes")
    digest = hashlib.sha256(quote.encode("utf-8")).hexdigest()[:10]
    candidates.extend(f"{marker}{digest}{marker}" for marker in ("§", "¤", "※", "⌁", "◊", "☷"))
    for candidate in candidates:
        if candidate != quote and quote not in candidate:
            return candidate
    raise ValueError("could not construct a disjoint deterministic replacement")


def _pointer_signature(pointer: SpanPointer) -> tuple[str, int, int, str]:
    return pointer.field, pointer.doc, pointer.sent, pointer.quote


def _replace_pointer(pointer: SpanPointer, target: SpanPointer, value: str) -> SpanPointer:
    if _pointer_signature(pointer) != _pointer_signature(target):
        return pointer
    return replace(pointer, start=-1, end=-1, quote=value)


def _repair_program(program: Program, target: SpanPointer, value: str) -> Program:
    answer = program.answer
    pointer = (
        _replace_pointer(answer.pointer, target, value)
        if answer.pointer is not None
        else None
    )
    repaired_answer = replace(
        answer,
        pointer=pointer,
        operands=tuple(_replace_pointer(item, target, value) for item in answer.operands),
        labels=tuple(_replace_pointer(item, target, value) for item in answer.labels),
    )
    repaired_joins = tuple(
        replace(join, left=_replace_pointer(join.left, target, value))
        for join in program.joins
    )
    return replace(program, answer=repaired_answer, joins=repaired_joins)


def _value_pointer(example: Example, program: Program) -> SpanPointer | None:
    answer = program.answer
    candidates = ([answer.pointer] if answer.pointer is not None else []) + list(answer.operands)
    join_pointers = {_pointer_signature(join.left) for join in program.joins}
    candidates.sort(key=lambda item: _pointer_signature(item) in join_pointers)
    for pointer in candidates:
        if pointer.quote and pointer.field != "question" and pointer.quote in _source(example, pointer):
            return pointer
    return None


def evaluate_example_updates(example: Example, program: Program) -> dict[str, dict[str, object]]:
    original = execute(example, program)
    if not original.lineage_certified:
        return {}
    snapshot = bind_lineage(example, program)
    counts = {
        "unread_cell_update": ActionCounts(expected_action="reuse"),
        "value_preserving_rewrite": ActionCounts(expected_action="replay"),
        "answer_value_replacement": ActionCounts(expected_action="recompile"),
        "sentence_insertion": ActionCounts(expected_action="recompile"),
        "sentence_deletion": ActionCounts(expected_action="recompile"),
    }

    dependencies = {_cell_key(cell) for cell in snapshot.cells if cell.field != "question"}
    unread = [
        cell
        for cell in _all_context_cells(example)
        if _cell_key(cell) not in dependencies and cell.field == "sentence"
    ]
    if unread:
        updated = _append_to_cell(example, unread[0], " [unread cell updated]")
        action = refresh_result(updated, program, original.answer, snapshot).action
        counts["unread_cell_update"].record(action)

    read_cells = [cell for cell in snapshot.cells if cell.field != "question"]
    if read_cells:
        updated = _append_to_cell(example, read_cells[0], " [wording updated]")
        action = refresh_result(updated, program, original.answer, snapshot).action
        counts["value_preserving_rewrite"].record(action)

    pointer = _value_pointer(example, program)
    if pointer is not None:
        value = _replacement(pointer.quote)
        updated = _replace_source(example, pointer, pointer.quote, value)
        action = refresh_result(updated, program, original.answer, snapshot).action
        metric = counts["answer_value_replacement"]
        metric.record(action)
        repaired = execute(updated, _repair_program(program, pointer, value))
        metric.oracle_repair_valid += int(repaired.lineage_certified)
        metric.oracle_answer_changed += int(
            repaired.lineage_certified and repaired.answer != original.answer
        )

    evidence_docs = sorted({doc for doc, _ in program.evidence})
    if evidence_docs:
        doc = evidence_docs[0]
        documents = list(example.documents)
        document = documents[doc]
        inserted = "Newly inserted potentially relevant fact."
        documents[doc] = replace(document, sentences=document.sentences + (inserted,))
        updated = replace(example, documents=tuple(documents))
        action = refresh_result(updated, program, original.answer, snapshot).action
        counts["sentence_insertion"].record(action)

    evidence_sentences = sorted(program.evidence)
    if evidence_sentences:
        doc, sent = evidence_sentences[-1]
        documents = list(example.documents)
        document = documents[doc]
        sentences = list(document.sentences)
        sentences.pop(sent)
        documents[doc] = replace(document, sentences=tuple(sentences))
        updated = replace(example, documents=tuple(documents))
        action = refresh_result(updated, program, original.answer, snapshot).action
        counts["sentence_deletion"].record(action)

    return {name: metric.to_dict() for name, metric in counts.items()}


def evaluate_update_stress(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    limit: int | None = None,
    dataset_variant: str | None = None,
) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["id"]): row for row in rows}
    aggregate = {
        "unread_cell_update": ActionCounts(expected_action="reuse"),
        "value_preserving_rewrite": ActionCounts(expected_action="replay"),
        "answer_value_replacement": ActionCounts(expected_action="recompile"),
        "sentence_insertion": ActionCounts(expected_action="recompile"),
        "sentence_deletion": ActionCounts(expected_action="recompile"),
    }
    examples = valid = certified = 0
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
        valid += int(result.valid)
        if not result.lineage_certified:
            continue
        certified += 1
        local = evaluate_example_updates(example, program)
        for name, payload in local.items():
            target = aggregate[name]
            for field in (
                "eligible",
                "reuse",
                "replay",
                "recompile",
                "expected_action_hits",
                "oracle_repair_valid",
                "oracle_answer_changed",
            ):
                setattr(target, field, getattr(target, field) + int(payload[field]))
    return {
        "dataset": dataset,
        "dataset_variant": dataset_variant
        or ("distractor" if dataset == "hotpotqa" else "default"),
        "examples": examples,
        "valid_programs": valid,
        "certified_programs": certified,
        "update_classes": {name: value.to_dict() for name, value in aggregate.items()},
        "scope": (
            "deterministic counterfactual updates applied to every eligible lineage-certified "
            "program; repaired-program results are oracle diagnostics and never used for routing"
        ),
        "safety_policy": (
            "content updates use exact read-cell hashes; any insertion or deletion changes the "
            "candidate-context cell-ID domain and conservatively requests recompilation"
        ),
    }
