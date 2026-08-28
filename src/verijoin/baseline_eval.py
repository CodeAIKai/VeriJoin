from __future__ import annotations

import json
import re
from pathlib import Path

from .data import iter_examples
from .text import exact_match, token_f1

_ANSWER = re.compile(r"<ANSWER>\s*(.*?)\s*</ANSWER>", flags=re.DOTALL)
_CITATION = re.compile(r"<CITATION>\s*(\{.*?\})\s*</CITATION>", flags=re.DOTALL)


def _set_f1(predicted: set[tuple[int, int]], gold: set[tuple[int, int]]) -> float:
    if not predicted and not gold:
        return 1.0
    overlap = len(predicted & gold)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def evaluate_output_baseline(
    dataset: str,
    raw_root: Path,
    split: str,
    predictions: Path,
    *,
    task: str,
    limit: int | None = None,
    dataset_variant: str | None = None,
) -> dict[str, object]:
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    total = parsed = valid_citations = 0
    em_sum = f1_sum = evidence_f1_sum = 0.0
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        if limit is not None and total >= limit:
            break
        total += 1
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        output = str(row.get("output", ""))
        answer = ""
        evidence: set[tuple[int, int]] = set()
        try:
            if task == "answer":
                match = _ANSWER.search(output)
                if match is None:
                    continue
                answer = match.group(1).strip()
            else:
                match = _CITATION.search(output)
                if match is None:
                    continue
                payload = json.loads(match.group(1))
                answer = str(payload["answer"])
                evidence = {(int(ref[0]), int(ref[1])) for ref in payload["evidence"]}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        parsed += 1
        em_sum += exact_match(answer, example.answers)
        f1_sum += token_f1(answer, example.answers)
        if task == "citation":
            citations_valid = all(
                0 <= doc < len(example.documents)
                and 0 <= sent < len(example.documents[doc].sentences)
                for doc, sent in evidence
            )
            valid_citations += int(citations_valid)
            gold = set(example.support)
            evidence_f1_sum += _set_f1(evidence, gold) if example.support_complete else 0.0
    denominator = total or 1
    return {
        "dataset": dataset,
        "dataset_variant": dataset_variant
        or ("distractor" if dataset == "hotpotqa" else "default"),
        "task": task,
        "examples": total,
        "predictions": len(rows),
        "parse_rate": 100.0 * parsed / denominator,
        "answer_em": 100.0 * em_sum / denominator,
        "answer_f1": 100.0 * f1_sum / denominator,
        "valid_citation_rate": 100.0 * valid_citations / denominator
        if task == "citation"
        else None,
        "evidence_f1": 100.0 * evidence_f1_sum / denominator
        if task == "citation"
        else None,
    }
