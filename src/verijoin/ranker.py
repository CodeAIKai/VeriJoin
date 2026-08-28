from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import iter_examples
from .schema import Example, Program
from .text import exact_match, normalize_answer
from .vm import execute, parse_program


@dataclass(frozen=True, slots=True)
class RankCandidate:
    index: int
    output: str
    answer: str
    tier: int
    passage: str
    average_logprob: float


@dataclass(frozen=True, slots=True)
class RankerBuildSummary:
    datasets: tuple[str, ...]
    questions_seen: int
    questions_with_pairs: int
    train_pairs: int
    eval_pairs: int
    train_output: str
    eval_output: str
    train_sha256: str
    eval_sha256: str


@dataclass(frozen=True, slots=True)
class RankerSelectionSummary:
    dataset: str
    dataset_variant: str
    examples: int
    predictions: int
    ranked_candidates: int
    model_seconds: float
    examples_per_second: float
    output: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cited_evidence(example: Example, program: Program) -> tuple[str, ...]:
    evidence: list[str] = []
    seen: set[tuple[int, int]] = set()
    for doc_id, sent_id in program.evidence:
        ref = (doc_id, sent_id)
        if ref in seen or not (0 <= doc_id < len(example.documents)):
            continue
        document = example.documents[doc_id]
        if not (0 <= sent_id < len(document.sentences)):
            continue
        seen.add(ref)
        evidence.append(f"[{document.title}] {document.sentences[sent_id]}")
    return tuple(evidence)


def _passage(
    answer: str,
    evidence: tuple[str, ...],
    tier: int,
    agreement: int,
    candidate_count: int,
) -> str:
    status = "executable" if tier == 2 else "answer-only"
    cited = "\n".join(evidence) if evidence else "(no valid cited sentence)"
    return (
        f"Candidate answer: {answer}\n"
        f"Program status: {status}\n"
        f"Answer agreement: {agreement} of {candidate_count}\n"
        f"Cited evidence:\n{cited}"
    )


def candidate_records(example: Example, row: dict[str, object]) -> list[RankCandidate]:
    stored = row.get("candidates")
    if not isinstance(stored, list) or not stored:
        raise ValueError(f"{example.qid} has no stored candidates")
    prepared: list[tuple[int, str, str, int, tuple[str, ...], float]] = []
    for index, candidate in enumerate(stored):
        if not isinstance(candidate, dict):
            continue
        raw = str(candidate.get("output", ""))
        try:
            program = parse_program(raw)
            result = execute(example, program)
        except (ValueError, IndexError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if not result.answer_valid:
            continue
        prepared.append(
            (
                index,
                raw,
                result.candidate_answer,
                2 if result.valid else 1,
                _cited_evidence(example, program),
                float(candidate.get("average_logprob", float("-inf"))),
            )
        )
    counts = Counter(normalize_answer(item[2]) for item in prepared)
    return [
        RankCandidate(
            index=index,
            output=raw,
            answer=answer,
            tier=tier,
            passage=_passage(
                answer,
                evidence,
                tier,
                counts[normalize_answer(answer)],
                len(stored),
            ),
            average_logprob=average_logprob,
        )
        for index, raw, answer, tier, evidence, average_logprob in prepared
    ]


def _holdout(dataset: str, qid: str, fraction: float) -> bool:
    value = int.from_bytes(hashlib.sha256(f"ranker:{dataset}:{qid}".encode()).digest()[:8], "big")
    return value < int(fraction * 2**64)


def build_ranker_pairs(
    datasets: tuple[str, ...],
    raw_root: Path,
    candidates_dir: Path,
    train_output: Path,
    eval_output: Path,
    *,
    holdout_fraction: float = 0.1,
) -> RankerBuildSummary:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    questions_seen = questions_with_pairs = 0
    for dataset in datasets:
        path = candidates_dir / f"stage3-{dataset}-train-n4.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_id = {str(row["id"]): row for row in rows}
        for example in iter_examples(dataset, raw_root, "train"):
            row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
            if row is None:
                continue
            questions_seen += 1
            records = candidate_records(example, row)
            positives = [
                record for record in records if exact_match(record.answer, example.answers)
            ]
            negatives = [
                record for record in records if not exact_match(record.answer, example.answers)
            ]
            if not positives or not negatives:
                continue
            questions_with_pairs += 1
            target = eval_rows if _holdout(dataset, example.qid, holdout_fraction) else train_rows
            seen_pairs: set[tuple[str, str]] = set()
            for positive in positives:
                for negative in negatives:
                    pair = (positive.passage, negative.passage)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    target.append(
                        {
                            "id": example.qid,
                            "dataset": dataset,
                            "query": example.question,
                            "positive": positive.passage,
                            "negative": negative.passage,
                        }
                    )
    for path, rows in ((train_output, train_rows), (eval_output, eval_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
    return RankerBuildSummary(
        datasets=datasets,
        questions_seen=questions_seen,
        questions_with_pairs=questions_with_pairs,
        train_pairs=len(train_rows),
        eval_pairs=len(eval_rows),
        train_output=str(train_output),
        eval_output=str(eval_output),
        train_sha256=_file_sha256(train_output),
        eval_sha256=_file_sha256(eval_output),
    )


def _select_ranked(records: list[RankCandidate], scores: list[float]) -> int:
    if len(records) != len(scores) or not records:
        raise ValueError("records and scores must have the same non-zero length")
    groups: dict[str, list[tuple[RankCandidate, float]]] = defaultdict(list)
    for record, score in zip(records, scores):
        groups[normalize_answer(record.answer)].append((record, score))
    best_answer = max(
        groups,
        key=lambda answer: (
            max(score for _, score in groups[answer]),
            len(groups[answer]),
            answer,
        ),
    )
    best = max(
        groups[best_answer],
        key=lambda item: (item[0].tier, item[1], -item[0].index),
    )
    return best[0].index


def reselect_with_ranker(
    dataset: str,
    raw_root: Path,
    split: str,
    source: Path,
    output: Path,
    model_path: str,
    adapter_path: str | None,
    *,
    batch_size: int = 64,
    max_length: int = 512,
    dataset_variant: str | None = None,
) -> RankerSelectionSummary:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.cuda().eval()

    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    groups: list[tuple[Example, dict[str, Any], list[RankCandidate]]] = []
    flat: list[tuple[str, str]] = []
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        records = candidate_records(example, row)
        groups.append((example, row, records))
        flat.extend((example.question, record.passage) for record in records)

    scores: list[float] = []
    model_start = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            encoded = tokenizer(
                flat[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to("cuda")
            scores.extend(model(**encoded, return_dict=True).logits.view(-1).float().cpu().tolist())
    model_seconds = time.perf_counter() - model_start

    derived_rows: list[dict[str, Any]] = []
    offset = 0
    ranked_candidates = 0
    for _, row, records in groups:
        local_scores = scores[offset : offset + len(records)]
        offset += len(records)
        ranked_candidates += len(records)
        derived = dict(row)
        stored = [dict(candidate) for candidate in row.get("candidates", [])]
        for record, score in zip(records, local_scores):
            stored[record.index]["ranker_score"] = score
        derived["candidates"] = stored
        if records:
            selected_index = _select_ranked(records, local_scores)
            chosen = stored[selected_index]
            derived.update(
                {
                    "output": str(chosen["output"]),
                    "finish_reason": str(chosen["finish_reason"]),
                    "generated_tokens": int(chosen["generated_tokens"]),
                    "selection": {
                        "kind": "train_only_semantic_ranker",
                        "selected_index": selected_index,
                        "answer_valid_candidates": len(records),
                        "strict_valid_candidates": sum(record.tier == 2 for record in records),
                        "selected_ranker_score": stored[selected_index]["ranker_score"],
                    },
                }
            )
        derived_rows.append(derived)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in derived_rows
        ),
        encoding="utf-8",
    )
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(
            {
                "derived_from": str(source),
                "dataset": dataset,
                "dataset_variant": dataset_variant,
                "split": split,
                "ranker_model": model_path,
                "ranker_adapter": adapter_path,
                "batch_size": batch_size,
                "max_length": max_length,
                "uses_gold_labels": False,
                "status": "complete",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return RankerSelectionSummary(
        dataset=dataset,
        dataset_variant=dataset_variant or ("distractor" if dataset == "hotpotqa" else "default"),
        examples=len(groups),
        predictions=len(derived_rows),
        ranked_candidates=ranked_candidates,
        model_seconds=model_seconds,
        examples_per_second=len(groups) / model_seconds if model_seconds else 0.0,
        output=str(output),
    )


def build_summary_dict(summary: RankerBuildSummary) -> dict[str, object]:
    return asdict(summary)


def selection_summary_dict(summary: RankerSelectionSummary) -> dict[str, object]:
    return asdict(summary)
