from __future__ import annotations

import json
import re
from typing import Any

from .operators import canonical_scalar, ordered_scalar
from .schema import (
    AnswerExpr,
    Example,
    JoinConstraint,
    Program,
    SpanPointer,
    ValidationResult,
)
from .text import normalize_answer

_PROGRAM = re.compile(r"<PROGRAM>\s*(\{.*?\})\s*</PROGRAM>", flags=re.DOTALL)
VM_SEMANTICS_VERSION = "verijoin-vm-2"


def _span(value: dict[str, Any]) -> SpanPointer:
    return SpanPointer(
        doc=int(value["doc"]),
        sent=int(value["sent"]),
        start=int(value.get("start", -1)),
        end=int(value.get("end", -1)),
        field=str(value.get("field", "sentence")),  # type: ignore[arg-type]
        quote=str(value.get("quote", "")),
    )


def program_from_dict(value: dict[str, Any]) -> Program:
    if not isinstance(value, dict):
        raise TypeError("program must be a JSON object")
    answer_data = value["answer"]
    if not isinstance(answer_data, dict):
        raise TypeError("program.answer must be a JSON object")
    pointer_data = answer_data.get("pointer")
    answer = AnswerExpr(
        op=str(answer_data["op"]),  # type: ignore[arg-type]
        pointer=_span(pointer_data) if pointer_data else None,
        value=str(answer_data.get("value", "")),
        operands=tuple(_span(item) for item in answer_data.get("operands", [])),
        labels=tuple(_span(item) for item in answer_data.get("labels", [])),
        value_type=str(answer_data.get("value_type", "text")),  # type: ignore[arg-type]
    )
    joins = tuple(
        JoinConstraint(
            left=_span(item["left"]),
            right_doc=int(item["right_doc"]),
            right_field=str(item.get("right_field", "title")),  # type: ignore[arg-type]
            kind=str(item.get("kind", "equi")),  # type: ignore[arg-type]
        )
        for item in value.get("joins", [])
    )
    return Program(
        version=int(value.get("version", 1)),
        mode=str(value.get("mode", "unknown")),  # type: ignore[arg-type]
        evidence=tuple((int(ref[0]), int(ref[1])) for ref in value.get("evidence", [])),
        joins=joins,
        answer=answer,
        metadata=dict(value.get("metadata", {})),
    )


def parse_program(text: str) -> Program:
    match = _PROGRAM.search(text)
    payload = match.group(1) if match else text.strip()
    return program_from_dict(json.loads(payload))


def _read_span(example: Example, pointer: SpanPointer, errors: list[str], label: str) -> str:
    if pointer.field == "question":
        source = example.question
        if pointer.doc != -1 or pointer.sent != -1:
            errors.append(f"{label}: question pointer must use doc=sent=-1")
    else:
        source = ""
    if pointer.doc < 0 or pointer.doc >= len(example.documents):
        if pointer.field != "question":
            errors.append(f"{label}: document out of range")
            return ""
    else:
        document = example.documents[pointer.doc]
        if pointer.field == "title":
            source = document.title
        elif pointer.field == "sentence":
            if pointer.sent < 0 or pointer.sent >= len(document.sentences):
                errors.append(f"{label}: sentence out of range")
                return ""
            source = document.sentences[pointer.sent]
        else:
            errors.append(f"{label}: invalid pointer field")
            return ""
    if pointer.quote:
        start = source.find(pointer.quote)
        if start < 0:
            errors.append(f"{label}: quoted span is absent from source")
            return ""
        return source[start : start + len(pointer.quote)]
    if pointer.start < 0 or pointer.end <= pointer.start or pointer.end > len(source):
        errors.append(f"{label}: character span out of range")
        return ""
    return source[pointer.start : pointer.end]


def execute(
    example: Example,
    program: Program,
    *,
    allow_literal: bool = False,
    require_join_for_multidoc: bool = True,
    validate_joins: bool = True,
) -> ValidationResult:
    errors: list[str] = []
    evidence = tuple(dict.fromkeys(program.evidence))
    cited_text: list[str] = []
    for index, (doc, sent) in enumerate(evidence):
        if doc < 0 or doc >= len(example.documents):
            errors.append(f"evidence[{index}]: document out of range")
            continue
        if sent < 0 or sent >= len(example.documents[doc].sentences):
            errors.append(f"evidence[{index}]: sentence out of range")
            continue
        cited_text.append(example.documents[doc].sentences[sent])

    join_values: list[str] = []
    join_edges: set[tuple[int, int]] = set()
    evidence_docs = {doc for doc, _ in evidence}
    if validate_joins:
        for index, join in enumerate(program.joins):
            value = _read_span(example, join.left, errors, f"join[{index}].left")
            join_values.append(value)
            query_join = join.kind == "query" and join.left.field == "question"
            if join.kind == "query" and not query_join:
                errors.append(f"join[{index}]: query join must read from the question")
            if join.kind == "equi" and join.left.field == "question":
                errors.append(f"join[{index}]: equi join cannot read from the question")
            if not query_join and join.left.ref() not in evidence:
                errors.append(f"join[{index}]: left pointer is not cited evidence")
            if join.right_doc not in evidence_docs:
                errors.append(f"join[{index}]: right document is not cited evidence")
                continue
            if join.right_doc < 0 or join.right_doc >= len(example.documents):
                errors.append(f"join[{index}]: right document out of range")
                continue
            if join.right_field not in {"title", "sentence"}:
                errors.append(f"join[{index}]: invalid right field")
                continue
            right = example.documents[join.right_doc]
            target = right.title if join.right_field == "title" else f"{right.title} {right.text}"
            if value and normalize_answer(value) not in normalize_answer(target):
                errors.append(f"join[{index}]: anchor is absent from right input")
            elif (
                value
                and join.right_doc in evidence_docs
                and (query_join or join.left.ref() in evidence)
            ):
                left_node = -1 if query_join else join.left.doc
                join_edges.add((left_node, join.right_doc))

    if (
        validate_joins
        and require_join_for_multidoc
        and len(evidence_docs) > 1
        and program.mode
        in {
            "bridge",
            "composition",
        }
    ):
        # Gold annotations occasionally omit a lexical bridge. The compiler records
        # this as invalid coverage rather than silently pretending the proof joined.
        if not program.joins:
            errors.append("multi-document proof has no executable join")
        elif join_edges:
            graph: dict[int, set[int]] = {doc: set() for doc in evidence_docs}
            for left, right in join_edges:
                graph.setdefault(left, set()).add(right)
                graph.setdefault(right, set()).add(left)
            visited: set[int] = set()
            frontier = [next(iter(evidence_docs))]
            while frontier:
                doc = frontier.pop()
                if doc in visited:
                    continue
                visited.add(doc)
                frontier.extend(graph[doc] - visited)
            if not evidence_docs.issubset(visited):
                errors.append("multi-document proof join graph is disconnected")

    answer_errors: list[str] = []

    def read_answer_pointer(pointer: SpanPointer, label: str) -> str:
        value = _read_span(example, pointer, answer_errors, label)
        cited = (
            pointer.doc in evidence_docs if pointer.field == "title" else pointer.ref() in evidence
        )
        if not cited:
            answer_errors.append(f"{label}: pointer is not cited evidence")
        return value

    answer = ""
    if program.answer.op == "copy":
        if program.answer.pointer is None:
            answer_errors.append("copy answer has no pointer")
        else:
            answer = read_answer_pointer(program.answer.pointer, "answer")
    elif program.answer.op == "bool":
        answer = normalize_answer(program.answer.value)
        if answer not in {"yes", "no"}:
            answer_errors.append("boolean answer must be yes or no")
    elif program.answer.op == "literal":
        answer = program.answer.value
        if not allow_literal:
            answer_errors.append("ungrounded literal answer is disabled")
    elif program.answer.op in {"argmin", "argmax", "equal", "common"}:
        operands = [
            read_answer_pointer(pointer, f"answer.operand[{index}]")
            for index, pointer in enumerate(program.answer.operands)
        ]
        if len(operands) < 2:
            answer_errors.append(f"{program.answer.op} requires at least two grounded operands")
        elif program.answer.op in {"equal", "common"}:
            same = len({canonical_scalar(value) for value in operands}) == 1
            if program.answer.op == "equal":
                answer = "yes" if same else "no"
            elif same:
                answer = operands[0]
            else:
                answer_errors.append("common operands do not normalize to one value")
        else:
            labels = [
                read_answer_pointer(pointer, f"answer.label[{index}]")
                for index, pointer in enumerate(program.answer.labels)
            ]
            if len(labels) != len(operands):
                answer_errors.append("ordered comparison requires one grounded label per operand")
            parsed = [ordered_scalar(value) for value in operands]
            if any(value is None for value in parsed):
                answer_errors.append("ordered comparison contains an unparseable scalar")
            elif len(labels) == len(operands):
                key = min if program.answer.op == "argmin" else max
                selected = key(range(len(parsed)), key=lambda index: parsed[index])  # type: ignore[index]
                answer = labels[selected]
    else:
        answer_errors.append(f"unknown answer operator: {program.answer.op}")

    all_errors = tuple(errors + answer_errors)
    valid = not all_errors
    return ValidationResult(
        valid=valid,
        lineage_certified=valid
        and program.answer.op in {"copy", "argmin", "argmax", "equal", "common"},
        answer=answer if valid else "",
        candidate_answer=answer if not answer_errors else "",
        answer_valid=not answer_errors,
        errors=all_errors,
        cited_text=tuple(cited_text),
        join_values=tuple(join_values),
    )
