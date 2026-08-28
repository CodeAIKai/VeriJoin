from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from .operators import canonical_scalar, ordered_scalar, reducer_from_question
from .schema import AnswerExpr, Example, JoinConstraint, Program, SpanPointer
from .text import find_span, normalize_answer

_TRAILING_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
_TITLE_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
_GENERIC_TITLE_TOKENS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "and",
    "film",
    "song",
    "album",
    "team",
    "season",
    "school",
    "university",
    "college",
    "episode",
    "series",
}


def _title_aliases(title: str) -> tuple[str, ...]:
    """Conservative aliases that remain literal substrings of a source title."""
    base = _TRAILING_PARENTHETICAL.sub("", title).strip()
    aliases = [title, base]
    tokens = base.split()
    if 2 <= len(tokens) <= 4 and all(token[:1].isupper() for token in tokens):
        aliases.append(tokens[-1])
    return tuple(dict.fromkeys(value for value in aliases if len(value) >= 3))


def _title_fragments(title: str) -> tuple[str, ...]:
    """Literal title n-grams used only when they are re-found in another source."""
    tokens = _TITLE_TOKEN.findall(_TRAILING_PARENTHETICAL.sub("", title))
    values = list(_title_aliases(title))
    for width in range(min(4, len(tokens)), 1, -1):
        for start in range(len(tokens) - width + 1):
            piece = " ".join(tokens[start : start + width])
            if any(
                token.casefold() not in _GENERIC_TITLE_TOKENS
                for token in tokens[start : start + width]
            ):
                values.append(piece)
    for token in tokens:
        if (
            len(token) >= 4
            and token.casefold() not in _GENERIC_TITLE_TOKENS
            and token[:1].isupper()
        ):
            values.append(token)
    return tuple(dict.fromkeys(values))


def _best_evidence(example: Example) -> tuple[tuple[int, int], ...]:
    """Reduce paragraph-level MuSiQue labels to one anchored sentence per hop."""
    if example.dataset != "musique":
        return tuple(dict.fromkeys(example.support))

    by_doc: dict[int, list[int]] = defaultdict(list)
    for doc, sent in example.support:
        by_doc[doc].append(sent)
    step_answers: dict[int, list[str]] = defaultdict(list)
    for step in example.decomposition:
        if step.support_doc is not None and step.answer:
            step_answers[step.support_doc].append(step.answer)

    selected: list[tuple[int, int]] = []
    for doc, sentence_ids in by_doc.items():
        candidates = step_answers.get(doc, [])
        candidates.extend(example.answers)
        match = None
        for sent in sentence_ids:
            if find_span(example.documents[doc].sentences[sent], candidates):
                match = (doc, sent)
                break
        selected.append(match or (doc, sentence_ids[0]))
    return tuple(selected)


def _mode(example: Example) -> str:
    kind = example.question_type.casefold()
    question = example.question.casefold()
    if "comparison" in kind or any(
        phrase in question
        for phrase in (
            "which is older",
            "which was earlier",
            "which is larger",
            "which is longer",
            "who was born first",
            "are both",
            "same nationality",
            "younger",
            "older",
            "which came first",
            "who came first",
            "won more",
        )
    ):
        return "comparison"
    if len(example.decomposition) >= 3:
        return "composition"
    if len({doc for doc, _ in example.support}) >= 2:
        return "bridge"
    return "unknown"


def _pointer(
    example: Example,
    ref: tuple[int, int],
    values: Iterable[str],
    *,
    field: str = "sentence",
) -> SpanPointer | None:
    source = example.documents[ref[0]].title if field == "title" else example.sentence(ref).text
    match = find_span(source, values)
    if match is None:
        return None
    start, end, quote = match
    sent = -1 if field == "title" else ref[1]
    return SpanPointer(ref[0], sent, start, end, field, quote)  # type: ignore[arg-type]


def _answer_expr(example: Example, evidence: tuple[tuple[int, int], ...]) -> AnswerExpr:
    primary = example.answers[0] if example.answers else ""
    typed = _typed_answer_expr(example, evidence)
    if typed is not None:
        return typed
    if normalize_answer(primary) in {"yes", "no"}:
        return AnswerExpr("bool", value=normalize_answer(primary))
    for ref in reversed(evidence):
        pointer = _pointer(example, ref, example.answers)
        if pointer is not None:
            return AnswerExpr("copy", pointer=pointer)
    for doc in reversed(tuple(dict.fromkeys(ref[0] for ref in evidence))):
        pointer = _pointer(example, (doc, 0), example.answers, field="title")
        if pointer is not None:
            return AnswerExpr("copy", pointer=pointer)
    # This is retained only so compilation coverage can be measured. Strict VM
    # execution rejects it, preventing an ungrounded answer from being scored.
    return AnswerExpr("literal", value=primary)


def _title_pointer(example: Example, doc: int) -> SpanPointer:
    title = example.documents[doc].title
    return SpanPointer(doc, -1, 0, len(title), "title", title)


def _pointer_anywhere(
    example: Example, evidence: tuple[tuple[int, int], ...], value: str
) -> SpanPointer | None:
    for ref in evidence:
        pointer = _pointer(example, ref, (value,))
        if pointer is not None:
            return pointer
    return None


def _title_doc(example: Example, value: str, allowed_docs: set[int]) -> int | None:
    wanted = normalize_answer(value)
    for document in example.documents:
        if document.doc not in allowed_docs:
            continue
        aliases = {normalize_answer(alias) for alias in _title_aliases(document.title)}
        if wanted in aliases:
            return document.doc
    return None


def _root_subject(example: Example, subject: str) -> str:
    """Follow reverse evidence edges to the candidate entity named in the question."""
    current = subject
    for _ in range(4):
        predecessors = [
            left
            for left, _, right in example.evidences
            if normalize_answer(right) == normalize_answer(current)
        ]
        if not predecessors:
            break
        named = [
            value
            for value in predecessors
            if normalize_answer(value) in normalize_answer(example.question)
        ]
        current = named[0] if named else predecessors[0]
    return current


def _typed_from_triples(
    example: Example, evidence: tuple[tuple[int, int], ...]
) -> AnswerExpr | None:
    if len(example.evidences) < 2:
        return None
    primary = normalize_answer(example.answers[0])
    by_relation: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for subject, relation, value in example.evidences:
        if _pointer_anywhere(example, evidence, value) is not None:
            by_relation[normalize_answer(relation)].append((subject, value))
    groups = [items for items in by_relation.values() if len(items) >= 2]
    groups.sort(key=lambda items: len(items), reverse=True)
    for items in reversed(groups):
        pair = items[-2:]
        operands = tuple(_pointer_anywhere(example, evidence, value) for _, value in pair)
        if any(pointer is None for pointer in operands):
            continue
        grounded = tuple(pointer for pointer in operands if pointer is not None)
        if primary in {"yes", "no"}:
            derived = "yes" if len({canonical_scalar(value) for _, value in pair}) == 1 else "no"
            if derived == primary:
                return AnswerExpr("equal", operands=grounded)
            continue
        reducer = reducer_from_question(example.question)
        parsed = [ordered_scalar(value) for _, value in pair]
        if reducer is None or any(value is None for value in parsed):
            continue
        roots = [_root_subject(example, subject) for subject, _ in pair]
        evidence_docs = {doc for doc, _ in evidence}
        label_docs = [_title_doc(example, root, evidence_docs) for root in roots]
        if any(doc is None for doc in label_docs) or len(set(label_docs)) != len(label_docs):
            continue
        labels = tuple(_title_pointer(example, doc) for doc in label_docs if doc is not None)
        selected = (min if reducer == "argmin" else max)(
            range(len(parsed)), key=lambda index: parsed[index]
        )
        if normalize_answer(example.documents[labels[selected].doc].title) != primary:
            continue
        return AnswerExpr(reducer, operands=grounded, labels=labels, value_type="date")
    return None


def _typed_answer_expr(
    example: Example, evidence: tuple[tuple[int, int], ...]
) -> AnswerExpr | None:
    from_triples = _typed_from_triples(example, evidence)
    if from_triples is not None:
        return from_triples
    if not example.answers or normalize_answer(example.answers[0]) in {"yes", "no"}:
        return None
    pointers: list[SpanPointer] = []
    seen_docs: set[int] = set()
    for ref in evidence:
        if ref[0] in seen_docs:
            continue
        pointer = _pointer(example, ref, example.answers)
        if pointer is not None:
            pointers.append(pointer)
            seen_docs.add(ref[0])
    if len(pointers) >= 2 and len({canonical_scalar(pointer.quote) for pointer in pointers}) == 1:
        return AnswerExpr("common", operands=tuple(pointers))
    return None


def _join_candidates(example: Example) -> list[tuple[str, tuple[int, int]]]:
    candidates: list[tuple[str, tuple[int, int]]] = []
    support_docs = {doc for doc, _ in example.support}
    for step in example.decomposition:
        if step.support_doc is not None and step.answer:
            for sent in range(len(example.documents[step.support_doc].sentences)):
                candidates.append((step.answer, (step.support_doc, sent)))
    for subject, _, obj in example.evidences:
        for value in (subject, obj):
            for ref in example.support:
                if find_span(example.sentence(ref).text, (value,)):
                    candidates.append((value, ref))
                    break
    for document in example.documents:
        if document.doc in support_docs and document.title:
            aliases = _title_aliases(document.title)
            for ref in example.support:
                if ref[0] == document.doc:
                    continue
                match = find_span(example.sentence(ref).text, aliases)
                if match:
                    candidates.append((match[2], ref))
    return candidates


def _joins(example: Example, evidence: tuple[tuple[int, int], ...]) -> tuple[JoinConstraint, ...]:
    if _mode(example) == "comparison":
        return ()
    evidence_docs = {doc for doc, _ in evidence}
    best_by_pair: dict[tuple[int, int], JoinConstraint] = {}
    for value, left_ref in _join_candidates(example):
        if left_ref not in evidence:
            continue
        pointer = _pointer(example, left_ref, (value,))
        if pointer is None:
            continue
        for right_doc in sorted(evidence_docs - {left_ref[0]}):
            right = example.documents[right_doc]
            right_text = f"{right.title} {right.text}"
            if not find_span(right_text, (value,)):
                continue
            key = tuple(sorted((pointer.doc, right_doc)))
            candidate = JoinConstraint(pointer, right_doc, "sentence")
            previous = best_by_pair.get(key)
            if previous is None or len(pointer.quote) > len(previous.left.quote):
                best_by_pair[key] = candidate
            break
    for right_doc in sorted(evidence_docs):
        match = find_span(example.question, _title_fragments(example.documents[right_doc].title))
        if match is None:
            continue
        pointer = SpanPointer(-1, -1, match[0], match[1], "question", match[2])
        best_by_pair[(-1, right_doc)] = JoinConstraint(pointer, right_doc, "sentence", "query")
    for right_doc in sorted(evidence_docs):
        for left_ref in evidence:
            if left_ref[0] == right_doc:
                continue
            match = find_span(
                example.sentence(left_ref).text,
                _title_fragments(example.documents[right_doc].title),
            )
            if match is None:
                continue
            key = tuple(sorted((left_ref[0], right_doc)))
            candidate = JoinConstraint(
                SpanPointer(left_ref[0], left_ref[1], match[0], match[1], "sentence", match[2]),
                right_doc,
                "sentence",
                "equi",
            )
            previous = best_by_pair.get(key)
            if previous is None or len(candidate.left.quote) > len(previous.left.quote):
                best_by_pair[key] = candidate
            break
    return tuple(best_by_pair[key] for key in sorted(best_by_pair))


def compile_gold(example: Example) -> Program:
    if not example.answers or not example.support:
        raise ValueError(f"{example.dataset}/{example.qid} has no labeled proof")
    evidence = _best_evidence(example)
    answer = _answer_expr(example, evidence)
    joins = _joins(example, evidence)
    return Program(
        version=2,
        mode=_mode(example),  # type: ignore[arg-type]
        evidence=evidence,
        joins=joins,
        answer=answer,
        metadata={
            "dataset": example.dataset,
            "qid": example.qid,
            "unanchored_answer": answer.op == "literal",
            "join_count": len(joins),
        },
    )
