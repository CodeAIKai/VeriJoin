from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .compiler import compile_gold
from .data import iter_examples
from .text import exact_match, token_f1
from .vm import execute


@dataclass(frozen=True, slots=True)
class AuditSummary:
    dataset: str
    split: str
    examples: int
    compiled: int
    strict_valid: int
    anchored_answers: int
    joined_multidoc: int
    typed_answers: int
    answer_ops: dict[str, int]
    join_kinds: dict[str, int]
    oracle_em: float
    oracle_f1: float
    error_counts: dict[str, int]


def audit_dataset(dataset: str, root: Path, split: str, limit: int | None = None) -> AuditSummary:
    examples = compiled = valid = anchored = joined = 0
    em_total = f1_total = 0.0
    errors: Counter[str] = Counter()
    answer_ops: Counter[str] = Counter()
    join_kinds: Counter[str] = Counter()
    typed = 0
    for example in iter_examples(dataset, root, split):
        if limit is not None and examples >= limit:
            break
        examples += 1
        try:
            program = compile_gold(example)
            compiled += 1
        except (ValueError, IndexError, KeyError) as error:
            errors[f"compile:{type(error).__name__}"] += 1
            continue
        anchored += int(program.answer.op != "literal")
        joined += int(bool(program.joins))
        answer_ops[program.answer.op] += 1
        join_kinds.update(join.kind for join in program.joins)
        typed += int(program.answer.op in {"argmin", "argmax", "equal", "common"})
        result = execute(example, program)
        if result.valid:
            valid += 1
            em_total += exact_match(result.answer, example.answers)
            f1_total += token_f1(result.answer, example.answers)
        else:
            for error in result.errors:
                errors[error] += 1
    denominator = examples or 1
    return AuditSummary(
        dataset=dataset,
        split=split,
        examples=examples,
        compiled=compiled,
        strict_valid=valid,
        anchored_answers=anchored,
        joined_multidoc=joined,
        typed_answers=typed,
        answer_ops=dict(answer_ops.most_common()),
        join_kinds=dict(join_kinds.most_common()),
        oracle_em=100.0 * em_total / denominator,
        oracle_f1=100.0 * f1_total / denominator,
        error_counts=dict(errors.most_common()),
    )


def audit_dict(summary: AuditSummary) -> dict[str, object]:
    return asdict(summary)
