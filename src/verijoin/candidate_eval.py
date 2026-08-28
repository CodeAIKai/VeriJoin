from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import iter_examples
from .schema import Example
from .text import exact_match, normalize_answer, token_f1
from .vm import execute, parse_program


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    index: int
    tier: int
    answer: str
    answer_em: float
    answer_f1: float


@dataclass(frozen=True, slots=True)
class CandidateExampleAnalysis:
    candidates: int
    parsed_candidates: int
    answer_valid_candidates: int
    selected_em: float
    selected_f1: float
    oracle_em: float
    oracle_f1: float
    oracle_strict_em: float
    oracle_strict_f1: float
    global_consensus_em: float
    global_consensus_f1: float
    tier_consensus_em: float
    tier_consensus_f1: float


@dataclass(frozen=True, slots=True)
class CandidateAnalysisSummary:
    dataset: str
    dataset_variant: str
    examples: int
    predictions: int
    candidates: int
    candidate_parse_rate: float
    candidate_answer_valid_rate: float
    examples_with_answer_candidate_rate: float
    examples_with_correct_candidate_rate: float
    examples_with_strict_correct_candidate_rate: float
    selected_answer_em: float
    selected_answer_f1: float
    oracle_at_n_answer_em: float
    oracle_at_n_answer_f1: float
    oracle_at_n_strict_answer_em: float
    oracle_at_n_strict_answer_f1: float
    global_consensus_answer_em: float
    global_consensus_answer_f1: float
    tier_consensus_answer_em: float
    tier_consensus_answer_f1: float
    uses_gold_labels: bool = True


def _record(example: Example, index: int, raw: str) -> tuple[CandidateRecord | None, bool]:
    try:
        result = execute(example, parse_program(raw))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None, False
    if not result.answer_valid:
        return None, True
    tier = 2 if result.valid else 1
    return (
        CandidateRecord(
            index=index,
            tier=tier,
            answer=result.candidate_answer,
            answer_em=exact_match(result.candidate_answer, example.answers),
            answer_f1=token_f1(result.candidate_answer, example.answers),
        ),
        True,
    )


def _consensus(
    records: list[CandidateRecord], *, restrict_best_tier: bool
) -> CandidateRecord | None:
    if not records:
        return None
    eligible = records
    if restrict_best_tier:
        best_tier = max(record.tier for record in eligible)
        eligible = [record for record in eligible if record.tier == best_tier]
    counts = Counter(normalize_answer(record.answer) for record in eligible)
    best_answer = max(
        counts,
        key=lambda answer: (
            counts[answer],
            max(record.tier for record in eligible if normalize_answer(record.answer) == answer),
            -min(record.index for record in eligible if normalize_answer(record.answer) == answer),
            answer,
        ),
    )
    matches = [record for record in eligible if normalize_answer(record.answer) == best_answer]
    return max(matches, key=lambda record: (record.tier, -record.index))


def analyze_candidate_row(example: Example, row: dict[str, object]) -> CandidateExampleAnalysis:
    stored = row.get("candidates")
    if not isinstance(stored, list) or not stored:
        raise ValueError(f"{example.qid} has no stored candidates")
    records: list[CandidateRecord] = []
    parsed = 0
    for index, candidate in enumerate(stored):
        if not isinstance(candidate, dict):
            continue
        record, candidate_parsed = _record(example, index, str(candidate.get("output", "")))
        parsed += int(candidate_parsed)
        if record is not None:
            records.append(record)

    selected, _ = _record(example, -1, str(row.get("output", "")))
    global_consensus = _consensus(records, restrict_best_tier=False)
    tier_consensus = _consensus(records, restrict_best_tier=True)
    strict = [record for record in records if record.tier == 2]
    return CandidateExampleAnalysis(
        candidates=len(stored),
        parsed_candidates=parsed,
        answer_valid_candidates=len(records),
        selected_em=selected.answer_em if selected is not None else 0.0,
        selected_f1=selected.answer_f1 if selected is not None else 0.0,
        oracle_em=max((record.answer_em for record in records), default=0.0),
        oracle_f1=max((record.answer_f1 for record in records), default=0.0),
        oracle_strict_em=max((record.answer_em for record in strict), default=0.0),
        oracle_strict_f1=max((record.answer_f1 for record in strict), default=0.0),
        global_consensus_em=global_consensus.answer_em if global_consensus is not None else 0.0,
        global_consensus_f1=global_consensus.answer_f1 if global_consensus is not None else 0.0,
        tier_consensus_em=tier_consensus.answer_em if tier_consensus is not None else 0.0,
        tier_consensus_f1=tier_consensus.answer_f1 if tier_consensus is not None else 0.0,
    )


def analyze_candidates(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    dataset_variant: str | None = None,
) -> CandidateAnalysisSummary:
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    analyses: list[CandidateExampleAnalysis] = []
    total = 0
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        total += 1
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        analyses.append(analyze_candidate_row(example, row))

    denominator = total or 1
    candidate_count = sum(item.candidates for item in analyses)
    candidate_denominator = candidate_count or 1

    def percentage(field: str) -> float:
        return 100.0 * sum(float(getattr(item, field)) for item in analyses) / denominator

    return CandidateAnalysisSummary(
        dataset=dataset,
        dataset_variant=dataset_variant or ("distractor" if dataset == "hotpotqa" else "default"),
        examples=total,
        predictions=len(rows),
        candidates=candidate_count,
        candidate_parse_rate=100.0
        * sum(item.parsed_candidates for item in analyses)
        / candidate_denominator,
        candidate_answer_valid_rate=100.0
        * sum(item.answer_valid_candidates for item in analyses)
        / candidate_denominator,
        examples_with_answer_candidate_rate=100.0
        * sum(item.answer_valid_candidates > 0 for item in analyses)
        / denominator,
        examples_with_correct_candidate_rate=percentage("oracle_em"),
        examples_with_strict_correct_candidate_rate=percentage("oracle_strict_em"),
        selected_answer_em=percentage("selected_em"),
        selected_answer_f1=percentage("selected_f1"),
        oracle_at_n_answer_em=percentage("oracle_em"),
        oracle_at_n_answer_f1=percentage("oracle_f1"),
        oracle_at_n_strict_answer_em=percentage("oracle_strict_em"),
        oracle_at_n_strict_answer_f1=percentage("oracle_strict_f1"),
        global_consensus_answer_em=percentage("global_consensus_em"),
        global_consensus_answer_f1=percentage("global_consensus_f1"),
        tier_consensus_answer_em=percentage("tier_consensus_em"),
        tier_consensus_answer_f1=percentage("tier_consensus_f1"),
    )


def summary_dict(summary: CandidateAnalysisSummary) -> dict[str, object]:
    return asdict(summary)
