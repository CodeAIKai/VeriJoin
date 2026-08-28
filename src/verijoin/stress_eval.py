from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .data import iter_examples
from .lineage import SourceField, bind_lineage, verify_lineage
from .schema import Example, Program, SpanPointer
from .vm import execute, parse_program

CellKey = tuple[SourceField, int, int]


@dataclass(frozen=True, slots=True)
class AttackMetrics:
    attack: str
    eligible: int
    safe: int
    safe_rate: float
    lineage_response_rate: float | None
    execution_response_rate: float


def _answer_pointers(program: Program) -> tuple[SpanPointer, ...]:
    values = list(program.answer.operands) + list(program.answer.labels)
    if program.answer.pointer is not None:
        values.insert(0, program.answer.pointer)
    return tuple(values)


def _rewrite_cell(example: Example, key: CellKey, value: str) -> Example:
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


def _cell_text(example: Example, key: CellKey) -> str:
    field, doc, sent = key
    if field == "question":
        return example.question
    document = example.documents[doc]
    return document.title if field == "title" else document.sentences[sent]


def _counterfactual(example: Example, program: Program) -> Example | None:
    for pointer in _answer_pointers(program):
        key: CellKey = (pointer.field, pointer.doc, pointer.sent)
        source = _cell_text(example, key)
        if pointer.quote and pointer.quote in source:
            replacement = "Counterfactual Entity 99173"
            return _rewrite_cell(example, key, source.replace(pointer.quote, replacement, 1))
        if pointer.start >= 0 and pointer.end > pointer.start:
            replacement = "99173"
            return _rewrite_cell(
                example,
                key,
                source[: pointer.start] + replacement + source[pointer.end :],
            )
    return None


def _missing_source(example: Example, program: Program) -> Example | None:
    sentence_refs = sorted(program.evidence, key=lambda value: (value[0], value[1]), reverse=True)
    for doc, sent in sentence_refs:
        if 0 <= doc < len(example.documents) and 0 <= sent < len(example.documents[doc].sentences):
            documents = list(example.documents)
            document = documents[doc]
            documents[doc] = replace(document, sentences=document.sentences[:sent])
            return replace(example, documents=tuple(documents))
    return None


def _poison_distractor(
    example: Example, program: Program, dependency_keys: set[CellKey], answer: str
) -> Example | None:
    payload = " ".join(
        value for value in [answer, *(join.left.quote for join in program.joins)] if value
    )
    for document in reversed(example.documents):
        for sent in reversed(range(len(document.sentences))):
            key: CellKey = ("sentence", document.doc, sent)
            if key not in dependency_keys:
                source = document.sentences[sent]
                return _rewrite_cell(example, key, f"{source} Poisoned claim: {payload}.")
    return None


def evaluate_attacks(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    limit: int | None = None,
    dataset_variant: str | None = None,
) -> dict[str, object]:
    """Stress the certificate, not merely answer string accuracy.

    A counterfactual attack is safe for a cached result when the bound lineage
    detects staleness; fresh execution rejection is reported separately because
    another occurrence of the quoted value can still witness the same program.
    A missing-source attack requires both stale-lineage detection and fresh
    execution rejection. Distractor poisoning is safe when it does not invalidate
    or change an already bound proof. Disconnection is safe when a fresh execution
    rejects the edited proof graph.
    """
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    names = (
        "counterfactual_entity_or_number",
        "disconnected_evidence",
        "distractor_poisoning",
        "missing_source",
    )
    counts = {name: [0, 0, 0] for name in names}
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
        original = execute(example, program)
        if not original.valid:
            continue
        valid += 1
        snapshot = bind_lineage(example, program)
        if snapshot.lineage_certified:
            certified += 1
            changed = _counterfactual(example, program)
            if changed is not None:
                counts[names[0]][0] += 1
                stale = not verify_lineage(changed, snapshot).current
                rejected = not execute(changed, program).valid
                counts[names[0]][1] += int(stale)
                counts[names[0]][2] += int(rejected)

        evidence_docs = {doc for doc, _ in program.evidence}
        if program.mode in {"bridge", "composition"} and len(evidence_docs) > 1:
            counts[names[1]][0] += 1
            disconnected = replace(program, joins=())
            rejected = not execute(example, disconnected).valid
            counts[names[1]][1] += int(rejected)
            counts[names[1]][2] += int(rejected)

        dependency_keys = {(cell.field, cell.doc, cell.sent) for cell in snapshot.cells}
        poisoned = _poison_distractor(example, program, dependency_keys, original.answer)
        if poisoned is not None:
            counts[names[2]][0] += 1
            current = verify_lineage(poisoned, snapshot).current
            replay = execute(poisoned, program)
            replay_safe = replay.valid and replay.answer == original.answer
            counts[names[2]][1] += int(current)
            counts[names[2]][2] += int(replay_safe)

        missing = _missing_source(example, program)
        if missing is not None:
            counts[names[3]][0] += 1
            stale = not verify_lineage(missing, snapshot).current
            rejected = not execute(missing, program).valid
            counts[names[3]][1] += int(stale)
            counts[names[3]][2] += int(rejected)

    attacks = []
    for name, values in counts.items():
        denominator = values[0] or 1
        safe = min(values[1], values[2]) if name != names[0] else values[1]
        attacks.append(
            AttackMetrics(
                name,
                values[0],
                safe,
                100.0 * safe / denominator,
                None if name == names[1] else 100.0 * values[1] / denominator,
                100.0 * values[2] / denominator,
            )
        )
    return {
        "dataset": dataset,
        "dataset_variant": dataset_variant
        or ("distractor" if dataset == "hotpotqa" else "default"),
        "examples": examples,
        "valid_programs": valid,
        "lineage_certified_programs": certified,
        "scope": "post-generation certificate attacks",
        "attacks": [asdict(item) for item in attacks],
    }
