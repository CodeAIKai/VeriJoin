from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import iter_examples
from .text import token_f1
from .vm import execute, parse_program


def _new_bucket() -> dict[str, float]:
    return {
        "count": 0,
        "answer_valid": 0,
        "valid": 0,
        "certified": 0,
        "answer_f1_sum": 0.0,
        "strict_f1_sum": 0.0,
    }


def _finish_bucket(values: dict[str, float], parsed: int) -> dict[str, float | int]:
    count = int(values["count"])
    denominator = count or 1
    return {
        "count": count,
        "share_of_parsed": 100.0 * count / (parsed or 1),
        "answer_valid_rate": 100.0 * values["answer_valid"] / denominator,
        "valid_rate": 100.0 * values["valid"] / denominator,
        "certified_rate": 100.0 * values["certified"] / denominator,
        "answer_f1": 100.0 * values["answer_f1_sum"] / denominator,
        "strict_answer_f1": 100.0 * values["strict_f1_sum"] / denominator,
    }


def analyze_operator_coverage(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    limit: int | None = None,
    dataset_variant: str | None = None,
    allow_literal: bool = False,
) -> dict[str, Any]:
    """Break full-set accuracy and validity down by typed operator and join family."""
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    operators: defaultdict[str, dict[str, float]] = defaultdict(_new_bucket)
    joins: defaultdict[str, dict[str, float]] = defaultdict(_new_bucket)
    modes: defaultdict[str, dict[str, float]] = defaultdict(_new_bucket)
    total = parsed = valid = certified = 0
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        if limit is not None and total >= limit:
            break
        total += 1
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        try:
            program = parse_program(str(row.get("output", row.get("program", ""))))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        parsed += 1
        result = execute(example, program, allow_literal=allow_literal)
        kinds = {join.kind for join in program.joins}
        if kinds == {"query", "equi"}:
            join_family = "both"
        elif kinds == {"query"}:
            join_family = "query_only"
        elif kinds == {"equi"}:
            join_family = "equi_only"
        else:
            join_family = "none"
        valid += int(result.valid)
        certified += int(result.lineage_certified)
        answer_score = (
            token_f1(result.candidate_answer, example.answers) if result.answer_valid else 0.0
        )
        strict_score = token_f1(result.answer, example.answers) if result.valid else 0.0
        for bucket in (
            operators[program.answer.op],
            joins[join_family],
            modes[program.mode],
        ):
            bucket["count"] += 1
            bucket["answer_valid"] += int(result.answer_valid)
            bucket["valid"] += int(result.valid)
            bucket["certified"] += int(result.lineage_certified)
            bucket["answer_f1_sum"] += answer_score
            bucket["strict_f1_sum"] += strict_score
    return {
        "dataset": dataset,
        "dataset_variant": dataset_variant
        or ("distractor" if dataset == "hotpotqa" else "default"),
        "examples": total,
        "predictions": len(rows),
        "parsed": parsed,
        "parse_rate": 100.0 * parsed / (total or 1),
        "valid_rate": 100.0 * valid / (total or 1),
        "certified_rate": 100.0 * certified / (total or 1),
        "allow_literal": allow_literal,
        "by_answer_operator": {
            name: _finish_bucket(values, parsed)
            for name, values in sorted(operators.items())
        },
        "by_join_family": {
            name: _finish_bucket(values, parsed) for name, values in sorted(joins.items())
        },
        "by_mode": {
            name: _finish_bucket(values, parsed) for name, values in sorted(modes.items())
        },
    }
