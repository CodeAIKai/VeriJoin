from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .data import iter_examples
from .schema import Example, Program, SpanPointer
from .text import exact_match
from .update_stress import _repair_program, _replace_source, _value_pointer
from .vm import execute, parse_program

_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")
_YEAR = re.compile(r"^\d{4}$")
_RANGE_MARKS = ("-", "–", "—", "/")


@dataclass(frozen=True, slots=True)
class RecoveryCase:
    dataset: str
    qid: str
    example: Example
    old_program: Program
    repaired_program: Program
    old_answer: str
    new_answer: str
    pointer: SpanPointer
    old_quote: str
    new_quote: str


def _context_occurrences(example: Example, value: str) -> int:
    cells = [example.question]
    for document in example.documents:
        cells.append(document.title)
        cells.extend(document.sentences)
    return sum(cell.count(value) for cell in cells)


def mutate_numeric_quote(quote: str) -> str | None:
    """Make a small, plausible numeric/date mutation without changing other tokens."""
    matches = list(_NUMBER.finditer(quote))
    years = [match for match in matches if _YEAR.fullmatch(match.group().replace(",", ""))]
    if len(years) == 1 and not (
        len(matches) > 1 and any(mark in quote for mark in _RANGE_MARKS)
    ):
        target = years[0]
    elif len(matches) == 1:
        target = matches[0]
    else:
        return None
    raw = target.group()
    compact = raw.replace(",", "")
    if "." in compact:
        value = float(compact) + 1.0
        replacement = f"{value:.{len(compact.split('.', 1)[1])}f}"
    else:
        value = int(compact)
        if 29 <= value <= 31 or value == 2099:
            value -= 1
        else:
            value += 1
        replacement = f"{value:,}" if "," in raw else str(value)
    end = target.end()
    suffix = quote[end : end + 2]
    if suffix.casefold() in {"st", "nd", "rd", "th"} and "." not in compact:
        number = int(replacement.replace(",", ""))
        remainder = number % 100
        if 10 <= remainder <= 20:
            ordinal = "th"
        else:
            ordinal = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
        replacement += ordinal.upper() if suffix.isupper() else ordinal
        end += 2
    changed = quote[: target.start()] + replacement + quote[end:]
    return changed if changed != quote else None


def build_recovery_cases(
    dataset: str,
    raw_root: Path,
    predictions: Path,
    *,
    split: str = "dev",
    limit: int = 200,
    dataset_variant: str | None = None,
) -> list[RecoveryCase]:
    """Select deterministic, originally-correct copy programs and mutate their answer fact."""
    rows = {
        str(row["id"]): row
        for row in (
            json.loads(line)
            for line in predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    candidates: list[RecoveryCase] = []
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        row = rows.get(example.qid) or rows.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        try:
            program = parse_program(str(row.get("output", row.get("program", ""))))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        original = execute(example, program)
        if (
            program.answer.op != "copy"
            or not original.lineage_certified
            or not exact_match(original.answer, example.answers)
        ):
            continue
        pointer = _value_pointer(example, program)
        if pointer is None or pointer.field == "question":
            continue
        new_quote = mutate_numeric_quote(pointer.quote)
        if (
            new_quote is None
            or _context_occurrences(example, pointer.quote) != 1
            or _context_occurrences(example, new_quote) != 0
        ):
            continue
        try:
            updated = _replace_source(example, pointer, pointer.quote, new_quote)
            repaired_program = _repair_program(program, pointer, new_quote)
            repaired = execute(updated, repaired_program)
        except (IndexError, ValueError):
            continue
        if not repaired.lineage_certified or repaired.answer == original.answer:
            continue
        updated = replace(updated, answers=(repaired.answer,))
        candidates.append(
            RecoveryCase(
                dataset=dataset,
                qid=example.qid,
                example=updated,
                old_program=program,
                repaired_program=repaired_program,
                old_answer=original.answer,
                new_answer=repaired.answer,
                pointer=pointer,
                old_quote=pointer.quote,
                new_quote=new_quote,
            )
        )
    candidates.sort(
        key=lambda case: hashlib.sha256(
            f"{dataset}:{case.qid}".encode()
        ).hexdigest()
    )
    return candidates[:limit]
