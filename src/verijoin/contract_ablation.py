from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import iter_examples
from .text import exact_match, token_f1
from .vm import execute, parse_program


@dataclass(frozen=True, slots=True)
class ContractMetrics:
    variant: str
    examples: int
    parsed: int
    accepted: int
    accepted_rate: float
    answer_em: float
    answer_f1: float
    strict_answer_em: float
    strict_answer_f1: float


def _metrics(
    variant: str,
    total: int,
    parsed: int,
    accepted: int,
    answer_em: float,
    answer_f1: float,
    strict_em: float,
    strict_f1: float,
) -> ContractMetrics:
    denominator = total or 1
    return ContractMetrics(
        variant=variant,
        examples=total,
        parsed=parsed,
        accepted=accepted,
        accepted_rate=100.0 * accepted / denominator,
        answer_em=100.0 * answer_em / denominator,
        answer_f1=100.0 * answer_f1 / denominator,
        strict_answer_em=100.0 * strict_em / denominator,
        strict_answer_f1=100.0 * strict_f1 / denominator,
    )


def evaluate_contract_ablation(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    limit: int | None = None,
    dataset_variant: str | None = None,
) -> dict[str, object]:
    """Re-score identical programs under successively weaker proof contracts.

    This isolates validator effects. It is deliberately distinct from separately
    trained answer-only and citation-only models, which are evaluated elsewhere.
    """
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    names = ("answer_only", "citation_only", "no_connectivity", "free_literal", "full")
    totals = {
        name: {
            "accepted": 0,
            "em": 0.0,
            "f1": 0.0,
            "strict_em": 0.0,
            "strict_f1": 0.0,
        }
        for name in names
    }
    total = parsed = 0
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        if limit is not None and total >= limit:
            break
        total += 1
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        raw = str(row.get("output", row.get("program", "")))
        try:
            program = parse_program(raw)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        parsed += 1
        results = {
            "citation_only": execute(
                example,
                program,
                allow_literal=False,
                require_join_for_multidoc=False,
                validate_joins=False,
            ),
            "no_connectivity": execute(
                example,
                program,
                allow_literal=False,
                require_join_for_multidoc=False,
            ),
            "free_literal": execute(example, program, allow_literal=True),
            "full": execute(example, program),
        }
        full = results["full"]
        if full.answer_valid:
            em = exact_match(full.candidate_answer, example.answers)
            f1 = token_f1(full.candidate_answer, example.answers)
            bucket = totals["answer_only"]
            bucket["accepted"] += 1
            bucket["em"] += em
            bucket["f1"] += f1
            bucket["strict_em"] += em
            bucket["strict_f1"] += f1
        for name, result in results.items():
            if result.answer_valid:
                totals[name]["em"] += exact_match(result.candidate_answer, example.answers)
                totals[name]["f1"] += token_f1(result.candidate_answer, example.answers)
            if result.valid:
                totals[name]["accepted"] += 1
                totals[name]["strict_em"] += exact_match(result.answer, example.answers)
                totals[name]["strict_f1"] += token_f1(result.answer, example.answers)
    variants = [
        _metrics(
            name,
            total,
            parsed,
            int(totals[name]["accepted"]),
            totals[name]["em"],
            totals[name]["f1"],
            totals[name]["strict_em"],
            totals[name]["strict_f1"],
        )
        for name in names
    ]
    return {
        "dataset": dataset,
        "dataset_variant": dataset_variant
        or ("distractor" if dataset == "hotpotqa" else "default"),
        "predictions": len(rows),
        "note": "validator-only ablation on identical generated programs",
        "variants": [asdict(item) for item in variants],
    }
