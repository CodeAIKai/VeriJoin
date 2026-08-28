from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .data import iter_examples
from .text import token_f1
from .vm import execute, parse_program

PredictionTask = Literal[
    "program-answer",
    "program-strict",
    "program-certified",
    "program-literal",
    "answer",
    "citation",
]
ComparisonSubset = Literal["all", "candidate-eligible"]
_ANSWER = re.compile(r"<ANSWER>\s*(.*?)\s*</ANSWER>", flags=re.DOTALL)
_CITATION = re.compile(r"<CITATION>\s*(\{.*?\})\s*</CITATION>", flags=re.DOTALL)


def _load(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def _program_scores(example: Any, row: dict[str, Any] | None) -> tuple[float, float]:
    if row is None:
        return 0.0, 0.0
    try:
        program = parse_program(str(row.get("output", row.get("program", ""))))
        result = execute(example, program)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return 0.0, 0.0
    answer = token_f1(result.candidate_answer, example.answers) if result.answer_valid else 0.0
    strict = token_f1(result.answer, example.answers) if result.valid else 0.0
    return answer, strict


def _task_score(
    example: Any, row: dict[str, Any] | None, task: PredictionTask
) -> tuple[float, bool]:
    """Return per-example answer F1 and candidate-subset eligibility."""
    if row is None:
        return 0.0, task != "program-certified"
    output = str(row.get("output", row.get("program", "")))
    if task.startswith("program-"):
        try:
            program = parse_program(output)
            result = execute(example, program, allow_literal=task == "program-literal")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return 0.0, task != "program-certified"
        if task == "program-answer":
            answer = result.candidate_answer if result.answer_valid else ""
            return token_f1(answer, example.answers) if answer else 0.0, True
        if task == "program-certified":
            if not result.lineage_certified:
                return 0.0, False
            return token_f1(result.answer, example.answers), True
        answer = result.answer if result.valid else ""
        return token_f1(answer, example.answers) if answer else 0.0, True
    try:
        if task == "answer":
            match = _ANSWER.search(output)
            answer = match.group(1).strip() if match is not None else ""
        else:
            match = _CITATION.search(output)
            payload = json.loads(match.group(1)) if match is not None else {}
            answer = str(payload["answer"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        answer = ""
    return token_f1(answer, example.answers) if answer else 0.0, True


def _paired_bootstrap(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int = 128,
) -> dict[str, float | int]:
    if differences.size == 0:
        raise ValueError("paired bootstrap requires at least one example")
    generator = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        current = min(batch_size, remaining)
        indices = generator.integers(
            0,
            differences.size,
            size=(current, differences.size),
            dtype=np.int32,
        )
        means.append(differences[indices].mean(axis=1))
        remaining -= current
    bootstrapped = np.concatenate(means)
    lower, upper = np.quantile(bootstrapped, [0.025, 0.975])
    below = int(np.sum(bootstrapped <= 0.0))
    above = int(np.sum(bootstrapped >= 0.0))
    return {
        "examples": int(differences.size),
        "bootstrap_samples": samples,
        "mean_difference": 100.0 * float(differences.mean()),
        "ci95_lower": 100.0 * float(lower),
        "ci95_upper": 100.0 * float(upper),
        "two_sided_p": min(1.0, 2.0 * (min(below, above) + 1) / (samples + 1)),
        "candidate_wins": int(np.sum(differences > 0.0)),
        "candidate_losses": int(np.sum(differences < 0.0)),
        "ties": int(np.sum(differences == 0.0)),
    }


def compare_program_predictions(
    dataset: str,
    raw_root: Path,
    split: str,
    baseline: Path,
    candidate: Path,
    *,
    samples: int = 5000,
    seed: int = 20260825,
    limit: int | None = None,
    dataset_variant: str | None = None,
) -> dict[str, Any]:
    """Compare two stored program runs with paired full-example bootstrap intervals."""
    baseline_rows = _load(baseline)
    candidate_rows = _load(candidate)
    baseline_answer: list[float] = []
    baseline_strict: list[float] = []
    candidate_answer: list[float] = []
    candidate_strict: list[float] = []
    for index, example in enumerate(iter_examples(dataset, raw_root, split, dataset_variant)):
        if limit is not None and index >= limit:
            break
        baseline_row = baseline_rows.get(example.qid) or baseline_rows.get(
            f"{dataset}:{example.qid}"
        )
        candidate_row = candidate_rows.get(example.qid) or candidate_rows.get(
            f"{dataset}:{example.qid}"
        )
        answer, strict = _program_scores(example, baseline_row)
        baseline_answer.append(answer)
        baseline_strict.append(strict)
        answer, strict = _program_scores(example, candidate_row)
        candidate_answer.append(answer)
        candidate_strict.append(strict)
    answer_difference = np.asarray(candidate_answer) - np.asarray(baseline_answer)
    strict_difference = np.asarray(candidate_strict) - np.asarray(baseline_strict)
    return {
        "dataset": dataset,
        "dataset_variant": dataset_variant
        or ("distractor" if dataset == "hotpotqa" else "default"),
        "baseline": str(baseline),
        "candidate": str(candidate),
        "seed": seed,
        "scope": "paired example bootstrap; does not measure training-seed variance",
        "answer_f1": _paired_bootstrap(answer_difference, samples=samples, seed=seed),
        "strict_answer_f1": _paired_bootstrap(
            strict_difference, samples=samples, seed=seed + 1
        ),
    }


def compare_task_predictions(
    dataset: str,
    raw_root: Path,
    split: str,
    baseline: Path,
    candidate: Path,
    *,
    baseline_task: PredictionTask,
    candidate_task: PredictionTask,
    subset: ComparisonSubset = "all",
    samples: int = 5000,
    seed: int = 20260825,
    limit: int | None = None,
    dataset_variant: str | None = None,
) -> dict[str, Any]:
    """Paired-bootstrap answer F1 across different stored output protocols."""
    baseline_rows = _load(baseline)
    candidate_rows = _load(candidate)
    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    total = eligible = 0
    for index, example in enumerate(iter_examples(dataset, raw_root, split, dataset_variant)):
        if limit is not None and index >= limit:
            break
        total += 1
        baseline_row = baseline_rows.get(example.qid) or baseline_rows.get(
            f"{dataset}:{example.qid}"
        )
        candidate_row = candidate_rows.get(example.qid) or candidate_rows.get(
            f"{dataset}:{example.qid}"
        )
        candidate_score, candidate_eligible = _task_score(
            example, candidate_row, candidate_task
        )
        eligible += int(candidate_eligible)
        if subset == "candidate-eligible" and not candidate_eligible:
            continue
        baseline_score, _ = _task_score(example, baseline_row, baseline_task)
        baseline_scores.append(baseline_score)
        candidate_scores.append(candidate_score)
    baseline_values = np.asarray(baseline_scores)
    candidate_values = np.asarray(candidate_scores)
    differences = candidate_values - baseline_values
    return {
        "dataset": dataset,
        "dataset_variant": dataset_variant
        or ("distractor" if dataset == "hotpotqa" else "default"),
        "baseline": str(baseline),
        "baseline_task": baseline_task,
        "candidate": str(candidate),
        "candidate_task": candidate_task,
        "subset": subset,
        "total_examples": total,
        "candidate_eligible_examples": eligible,
        "candidate_eligibility_rate": 100.0 * eligible / (total or 1),
        "compared_examples": int(differences.size),
        "baseline_answer_f1": 100.0 * float(baseline_values.mean()),
        "candidate_answer_f1": 100.0 * float(candidate_values.mean()),
        "seed": seed,
        "scope": "paired example bootstrap; does not measure training-seed variance",
        "answer_f1": _paired_bootstrap(
            differences, samples=samples, seed=seed
        ),
    }
