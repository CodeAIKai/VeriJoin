from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import iter_examples
from .text import exact_match, token_prf
from .vm import execute, parse_program


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    dataset: str
    dataset_variant: str
    examples: int
    predictions: int
    parse_rate: float
    valid_rate: float
    lineage_certified_rate: float
    answer_candidate_rate: float
    answer_em: float
    answer_f1: float
    strict_answer_em: float
    strict_answer_f1: float
    valid_answer_em: float
    valid_answer_f1: float
    lineage_certified_answer_em: float
    lineage_certified_answer_f1: float
    evidence_granularity: str
    evidence_examples: int
    evidence_em: float
    evidence_precision: float
    evidence_recall: float
    evidence_f1: float
    joint_em: float | None
    joint_f1: float | None
    errors: dict[str, int]


def _set_prf(predicted: set[object], gold: set[object]) -> tuple[float, float, float, float]:
    """Exact match and P/R/F1 for an unordered evidence set."""
    exact = float(predicted == gold)
    if not predicted and not gold:
        return exact, 1.0, 1.0, 1.0
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    return exact, precision, recall, f1


def evaluate_predictions(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    allow_literal: bool = False,
    limit: int | None = None,
    dataset_variant: str | None = None,
) -> EvaluationSummary:
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    total = parsed = valid = lineage_certified = candidate_valid = evidence_examples = 0
    em_sum = f1_sum = strict_em_sum = strict_f1_sum = 0.0
    evidence_em_sum = evidence_precision_sum = evidence_recall_sum = evidence_f1_sum = 0.0
    joint_em_sum = joint_f1_sum = 0.0
    certified_em_sum = certified_f1_sum = 0.0
    errors: Counter[str] = Counter()
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        if limit is not None and total >= limit:
            break
        total += 1
        if dataset == "musique":
            gold_evidence: set[object] = {doc for doc, _ in example.support}
        else:
            gold_evidence = set(example.support)
        score_evidence = example.support_complete
        if score_evidence:
            evidence_examples += 1
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            errors["missing_prediction"] += 1
            if score_evidence:
                scores = _set_prf(set(), gold_evidence)
                evidence_em_sum += scores[0]
                evidence_precision_sum += scores[1]
                evidence_recall_sum += scores[2]
                evidence_f1_sum += scores[3]
            continue
        raw = str(row.get("output", row.get("program", "")))
        if isinstance(row.get("program"), dict):
            raw = json.dumps(row["program"], ensure_ascii=False)
        try:
            program = parse_program(raw)
            parsed += 1
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            errors["parse_error"] += 1
            if score_evidence:
                scores = _set_prf(set(), gold_evidence)
                evidence_em_sum += scores[0]
                evidence_precision_sum += scores[1]
                evidence_recall_sum += scores[2]
                evidence_f1_sum += scores[3]
            continue
        if dataset == "musique":
            predicted_evidence: set[object] = {doc for doc, _ in program.evidence}
        else:
            predicted_evidence = set(program.evidence)
        support_em = support_precision = support_recall = support_f1 = 0.0
        if score_evidence:
            support_em, support_precision, support_recall, support_f1 = _set_prf(
                predicted_evidence, gold_evidence
            )
            evidence_em_sum += support_em
            evidence_precision_sum += support_precision
            evidence_recall_sum += support_recall
            evidence_f1_sum += support_f1
        result = execute(example, program, allow_literal=allow_literal)
        answer_em = answer_precision = answer_recall = answer_f1 = 0.0
        if result.answer_valid:
            candidate_valid += 1
            answer_em = exact_match(result.candidate_answer, example.answers)
            answer_precision, answer_recall, answer_f1 = token_prf(
                result.candidate_answer, example.answers
            )
            em_sum += answer_em
            f1_sum += answer_f1
        if dataset != "musique" and score_evidence:
            joint_em_sum += answer_em * support_em
            joint_precision = answer_precision * support_precision
            joint_recall = answer_recall * support_recall
            if joint_precision + joint_recall:
                joint_f1_sum += (
                    2 * joint_precision * joint_recall / (joint_precision + joint_recall)
                )
        if not result.valid:
            for error in result.errors:
                errors[error] += 1
            continue
        valid += 1
        strict_em_sum += answer_em
        strict_f1_sum += answer_f1
        if result.lineage_certified:
            lineage_certified += 1
            certified_em_sum += answer_em
            certified_f1_sum += answer_f1
        if not answer_em:
            errors["valid_wrong_answer"] += 1
    denominator = total or 1
    valid_denominator = valid or 1
    certified_denominator = lineage_certified or 1
    evidence_denominator = evidence_examples or 1
    joint_available = dataset != "musique" and evidence_examples == total
    return EvaluationSummary(
        dataset=dataset,
        dataset_variant=dataset_variant or ("distractor" if dataset == "hotpotqa" else "default"),
        examples=total,
        predictions=len(rows),
        parse_rate=100.0 * parsed / denominator,
        valid_rate=100.0 * valid / denominator,
        lineage_certified_rate=100.0 * lineage_certified / denominator,
        answer_candidate_rate=100.0 * candidate_valid / denominator,
        answer_em=100.0 * em_sum / denominator,
        answer_f1=100.0 * f1_sum / denominator,
        strict_answer_em=100.0 * strict_em_sum / denominator,
        strict_answer_f1=100.0 * strict_f1_sum / denominator,
        valid_answer_em=100.0 * strict_em_sum / valid_denominator,
        valid_answer_f1=100.0 * strict_f1_sum / valid_denominator,
        lineage_certified_answer_em=100.0 * certified_em_sum / certified_denominator,
        lineage_certified_answer_f1=100.0 * certified_f1_sum / certified_denominator,
        evidence_granularity="document" if dataset == "musique" else "sentence",
        evidence_examples=evidence_examples,
        evidence_em=100.0 * evidence_em_sum / evidence_denominator,
        evidence_precision=100.0 * evidence_precision_sum / evidence_denominator,
        evidence_recall=100.0 * evidence_recall_sum / evidence_denominator,
        evidence_f1=100.0 * evidence_f1_sum / evidence_denominator,
        joint_em=100.0 * joint_em_sum / denominator if joint_available else None,
        joint_f1=100.0 * joint_f1_sum / denominator if joint_available else None,
        errors=dict(errors.most_common()),
    )


def summary_dict(summary: EvaluationSummary) -> dict[str, object]:
    return asdict(summary)
