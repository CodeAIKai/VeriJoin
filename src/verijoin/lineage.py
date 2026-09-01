from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

from .schema import Example, Program, SpanPointer
from .text import normalize_answer
from .vm import VM_SEMANTICS_VERSION, execute

SourceField = Literal["sentence", "title", "question"]


@dataclass(frozen=True, slots=True, order=True)
class SourceCell:
    field: SourceField
    doc: int
    sent: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LineageSnapshot:
    dataset: str
    qid: str
    vm_semantics_version: str
    program_sha256: str
    answer_sha256: str
    structure_sha256: str
    lineage_certified: bool
    cells: tuple[SourceCell, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LineageCheck:
    current: bool
    changed: tuple[tuple[SourceField, int, int], ...]
    missing: tuple[tuple[SourceField, int, int], ...]
    structure_changed: bool = False


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _program_digest(program: Program) -> str:
    payload = program.to_dict()
    payload.pop("metadata", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _digest(encoded)


def _structure_digest(example: Example) -> str:
    """Commit the stable cell-ID domain so insertions/deletions fail closed."""
    shape = [(document.doc, len(document.sentences)) for document in example.documents]
    encoded = json.dumps(shape, separators=(",", ":"))
    return _digest(encoded)


def _source(example: Example, field: SourceField, doc: int, sent: int) -> str:
    if field == "question":
        if doc != -1 or sent != -1:
            raise IndexError("question lineage must use doc=sent=-1")
        return example.question
    document = example.documents[doc]
    if field == "title":
        return document.title
    return document.sentences[sent]


def _answer_pointers(program: Program) -> tuple[SpanPointer, ...]:
    pointers = list(program.answer.operands) + list(program.answer.labels)
    if program.answer.pointer is not None:
        pointers.append(program.answer.pointer)
    return tuple(pointers)


def _right_witness(
    example: Example, doc: int, right_field: str, value: str
) -> tuple[SourceField, int, int] | None:
    """Return one source cell sufficient to prove the join membership predicate."""
    document = example.documents[doc]
    needle = normalize_answer(value)
    if not needle:
        return None
    if needle in normalize_answer(document.title):
        return "title", doc, -1
    if right_field == "title":
        return None
    for sent, sentence in enumerate(document.sentences):
        if needle in normalize_answer(sentence):
            return "sentence", doc, sent
    return None


def bind_lineage(example: Example, program: Program) -> LineageSnapshot:
    """Bind a successful logical plan to exact versions of every source cell it reads."""
    result = execute(example, program)
    if not result.valid:
        raise ValueError("cannot bind lineage for an invalid program: " + "; ".join(result.errors))
    keys: set[tuple[SourceField, int, int]] = {
        ("sentence", doc, sent) for doc, sent in program.evidence
    }
    for pointer in _answer_pointers(program):
        keys.add((pointer.field, pointer.doc, pointer.sent))
    for join in program.joins:
        keys.add((join.left.field, join.left.doc, join.left.sent))
        witness = _right_witness(
            example, join.right_doc, join.right_field, join.left.quote
        )
        if witness is not None:
            keys.add(witness)
        elif join.right_field == "title":
            keys.add(("title", join.right_doc, -1))
        else:
            # Conservative fallback for an invalid or cross-cell anchor.
            keys.add(("title", join.right_doc, -1))
            keys.update(
                ("sentence", join.right_doc, sent)
                for sent in range(len(example.documents[join.right_doc].sentences))
            )
    cells = tuple(
        SourceCell(field, doc, sent, _digest(_source(example, field, doc, sent)))
        for field, doc, sent in sorted(keys)
    )
    return LineageSnapshot(
        example.dataset,
        example.qid,
        VM_SEMANTICS_VERSION,
        _program_digest(program),
        _digest(result.answer),
        _structure_digest(example),
        result.lineage_certified,
        cells,
    )


def verify_lineage(example: Example, snapshot: LineageSnapshot) -> LineageCheck:
    """Fail closed when a bound question, title, or sentence is missing or bytewise changed."""
    changed: list[tuple[SourceField, int, int]] = []
    missing: list[tuple[SourceField, int, int]] = []
    if example.dataset != snapshot.dataset or example.qid != snapshot.qid:
        return LineageCheck(False, (), (("question", -1, -1),), False)
    structure_changed = _structure_digest(example) != snapshot.structure_sha256
    for cell in snapshot.cells:
        key = (cell.field, cell.doc, cell.sent)
        try:
            value = _source(example, *key)
        except IndexError:
            missing.append(key)
            continue
        if _digest(value) != cell.sha256:
            changed.append(key)
    return LineageCheck(
        not changed and not missing and not structure_changed,
        tuple(changed),
        tuple(missing),
        structure_changed,
    )


def verify_result_binding(snapshot: LineageSnapshot, program: Program, answer: str) -> bool:
    """Check that cached program and answer bytes are the ones committed by the snapshot."""
    return (
        snapshot.vm_semantics_version == VM_SEMANTICS_VERSION
        and snapshot.program_sha256 == _program_digest(program)
        and snapshot.answer_sha256 == _digest(answer)
    )
