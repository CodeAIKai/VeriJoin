from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from .operators import canonical_scalar, ordered_scalar
from .prompt import (
    ANSWER_SYSTEM_PROMPT,
    CITATION_SYSTEM_PROMPT,
    FREE_LITERAL_SYSTEM_PROMPT,
    render_target,
)
from .schema import AnswerExpr, Program
from .vm import program_from_dict

AblationTask = Literal["answer", "citation", "free_literal"]

@dataclass(frozen=True, slots=True)
class AblationDataSummary:
    task: str
    source: str
    output: str
    rows: int
    source_sha256: str
    output_sha256: str


def answer_from_program(program: Program) -> str:
    answer = program.answer
    if answer.op == "copy":
        if answer.pointer is None:
            raise ValueError("copy program has no pointer")
        return answer.pointer.quote
    if answer.op in {"bool", "literal"}:
        return answer.value
    operands = [pointer.quote for pointer in answer.operands]
    if answer.op == "common":
        if not operands:
            raise ValueError("common program has no operands")
        return operands[0]
    if answer.op == "equal":
        return "yes" if len({canonical_scalar(value) for value in operands}) == 1 else "no"
    if answer.op in {"argmin", "argmax"}:
        parsed = [ordered_scalar(value) for value in operands]
        if not parsed or any(value is None for value in parsed):
            raise ValueError("ordered program has an unparseable operand")
        if len(answer.labels) != len(parsed):
            raise ValueError("ordered program label count differs from operands")
        key = min if answer.op == "argmin" else max
        selected = key(range(len(parsed)), key=lambda index: parsed[index])
        return answer.labels[selected].quote
    raise ValueError(f"unsupported answer operator: {answer.op}")


def _target(task: AblationTask, program: Program) -> tuple[str, str]:
    answer = answer_from_program(program)
    if task == "answer":
        return ANSWER_SYSTEM_PROMPT, f"<ANSWER>{answer}</ANSWER>"
    if task == "citation":
        body = json.dumps(
            {"answer": answer, "evidence": [list(ref) for ref in program.evidence]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return CITATION_SYSTEM_PROMPT, f"<CITATION>{body}</CITATION>"
    literal = replace(
        program,
        answer=AnswerExpr("literal", value=answer),
    )
    return FREE_LITERAL_SYSTEM_PROMPT, render_target(literal)


def build_ablation_data(source: Path, output: Path, task: AblationTask) -> dict[str, object]:
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as input_handle, output.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            source_digest.update(line.encode())
            row = json.loads(line)
            program = program_from_dict(row["program"])
            system, target = _target(task, program)
            transformed = dict(row)
            transformed["messages"] = [
                {"role": "system", "content": system},
                row["messages"][1],
                {"role": "assistant", "content": target},
            ]
            transformed["ablation_task"] = task
            encoded = (
                json.dumps(transformed, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            output_handle.write(encoded)
            output_digest.update(encoded.encode())
            rows += 1
    summary = AblationDataSummary(
        task,
        str(source),
        str(output),
        rows,
        source_digest.hexdigest(),
        output_digest.hexdigest(),
    )
    return asdict(summary)
