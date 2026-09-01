#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from verijoin.infer import _ANSWER_OUTPUT, _average_logprob, _select_candidate
from verijoin.prompt import inference_messages
from verijoin.recompile_recovery import RecoveryCase, build_recovery_cases
from verijoin.text import exact_match, token_f1
from verijoin.vm import execute, parse_program


def _summary(cases: list[RecoveryCase], rows: list[dict[str, Any]]) -> dict[str, object]:
    parsed = valid = certified = candidate_em = strict_em = old_plan_valid = 0
    candidate_f1 = strict_f1 = cached_f1 = cached_em = 0.0
    by_id = {str(row["id"]): row for row in rows}
    for case in cases:
        cached_em += exact_match(case.old_answer, case.example.answers)
        cached_f1 += token_f1(case.old_answer, case.example.answers)
        old_plan_valid += int(execute(case.example, case.old_program).valid)
        row = by_id.get(case.qid)
        if row is None:
            continue
        try:
            program = parse_program(str(row["output"]))
            parsed += 1
            result = execute(case.example, program)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, IndexError):
            continue
        if result.answer_valid:
            candidate_em += int(exact_match(result.candidate_answer, case.example.answers))
            candidate_f1 += token_f1(result.candidate_answer, case.example.answers)
        if result.valid:
            valid += 1
            strict_em += int(exact_match(result.answer, case.example.answers))
            strict_f1 += token_f1(result.answer, case.example.answers)
            certified += int(result.lineage_certified)
    n = len(cases) or 1
    return {
        "cases": len(cases),
        "selection": "SHA-256 order over originally correct, lineage-certified copy programs with a mutable numeric/date answer",
        "mutation": "one deterministic +/-1 numeric or year edit; the old value occurs once and the new value is absent from the full context",
        "cached_old_answer_em": 100.0 * cached_em / n,
        "cached_old_answer_f1": 100.0 * cached_f1 / n,
        "old_plan_valid_rate_after_update": 100.0 * old_plan_valid / n,
        "recompile_parse_rate": 100.0 * parsed / n,
        "recompile_valid_rate": 100.0 * valid / n,
        "recompile_certified_rate": 100.0 * certified / n,
        "recompile_candidate_em": 100.0 * candidate_em / n,
        "recompile_candidate_f1": 100.0 * candidate_f1 / n,
        "recompile_strict_em": 100.0 * strict_em / n,
        "recompile_strict_f1": 100.0 * strict_f1 / n,
    }


def _answer_summary(cases: list[RecoveryCase], rows: list[dict[str, Any]]) -> dict[str, object]:
    parsed = answer_em = 0
    answer_f1 = 0.0
    by_id = {str(row["id"]): row for row in rows}
    for case in cases:
        row = by_id.get(case.qid)
        if row is None:
            continue
        match = _ANSWER_OUTPUT.search(str(row["output"]))
        if match is None or not match.group(1).strip():
            continue
        parsed += 1
        answer = match.group(1).strip()
        answer_em += int(exact_match(answer, case.example.answers))
        answer_f1 += token_f1(answer, case.example.answers)
    n = len(cases) or 1
    return {
        "cases": len(cases),
        "parse_rate": 100.0 * parsed / n,
        "answer_em": 100.0 * answer_em / n,
        "answer_f1": 100.0 * answer_f1 / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--hotpotqa-predictions", type=Path, required=True)
    parser.add_argument("--twowiki-predictions", type=Path, required=True)
    parser.add_argument("--musique-predictions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--answer-adapter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit-per-dataset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    paths = {
        "hotpotqa": args.hotpotqa_predictions,
        "2wiki": args.twowiki_predictions,
        "musique": args.musique_predictions,
    }
    cases: list[RecoveryCase] = []
    eligible_pool_by_dataset: dict[str, int] = {}
    for dataset, path in paths.items():
        pool = build_recovery_cases(
            dataset,
            args.raw_root,
            path,
            limit=10**9,
            dataset_variant="distractor" if dataset == "hotpotqa" else None,
        )
        eligible_pool_by_dataset[dataset] = len(pool)
        cases.extend(pool[: args.limit_per_dataset])
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            inference_messages(case.example), tokenize=False, add_generation_prompt=True
        )
        for case in cases
    ]
    lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
    if any(length > 6144 for length in lengths):
        raise ValueError(f"recovery prompt exceeds 6144 tokens: {max(lengths)}")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    start = time.perf_counter()
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        max_model_len=6656,
        gpu_memory_utilization=0.9,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        max_cpu_loras=2,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=4,
        temperature=0.4,
        top_p=0.95,
        max_tokens=384,
        seed=args.seed,
        logprobs=0,
    )
    generated = engine.generate(
        prompts,
        sampling_params=sampling,
        lora_request=LoRARequest("verijoin", 1, args.adapter),
        use_tqdm=True,
    )
    generation_seconds = time.perf_counter() - start
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cases_by_dataset: dict[str, list[RecoveryCase]] = defaultdict(list)
    for case, result in zip(cases, generated):
        candidates = [
            (
                candidate.text,
                _average_logprob(candidate.cumulative_logprob, len(candidate.token_ids)),
            )
            for candidate in result.outputs
        ]
        selected_index, selection = _select_candidate(
            case.example, candidates, execution_guided=True, task="program"
        )
        selected = result.outputs[selected_index]
        rows_by_dataset[case.dataset].append(
            {
                "id": case.qid,
                "dataset": case.dataset,
                "output": selected.text,
                "selection": selection,
                "old_answer": case.old_answer,
                "new_answer": case.new_answer,
                "mutation": {
                    "field": case.pointer.field,
                    "doc": case.pointer.doc,
                    "sent": case.pointer.sent,
                    "old_quote": case.old_quote,
                    "new_quote": case.new_quote,
                },
                "candidates": [
                    {
                        "output": candidate.text,
                        "average_logprob": candidates[index][1],
                        "generated_tokens": len(candidate.token_ids),
                    }
                    for index, candidate in enumerate(result.outputs)
                ],
            }
        )
        cases_by_dataset[case.dataset].append(case)
    answer_rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    answer_generation_seconds = 0.0
    if args.answer_adapter:
        answer_prompts = [
            tokenizer.apply_chat_template(
                inference_messages(case.example, task="answer"),
                tokenize=False,
                add_generation_prompt=True,
            )
            for case in cases
        ]
        answer_sampling = SamplingParams(
            n=4,
            temperature=0.4,
            top_p=0.95,
            max_tokens=64,
            seed=args.seed,
            logprobs=0,
        )
        answer_start = time.perf_counter()
        answer_generated = engine.generate(
            answer_prompts,
            sampling_params=answer_sampling,
            lora_request=LoRARequest("answer-only", 2, args.answer_adapter),
            use_tqdm=True,
        )
        answer_generation_seconds = time.perf_counter() - answer_start
        for case, result in zip(cases, answer_generated):
            candidates = [
                (
                    candidate.text,
                    _average_logprob(candidate.cumulative_logprob, len(candidate.token_ids)),
                )
                for candidate in result.outputs
            ]
            selected_index, selection = _select_candidate(
                case.example, candidates, execution_guided=True, task="answer"
            )
            selected = result.outputs[selected_index]
            answer_rows_by_dataset[case.dataset].append(
                {
                    "id": case.qid,
                    "dataset": case.dataset,
                    "output": selected.text,
                    "selection": selection,
                    "old_answer": case.old_answer,
                    "new_answer": case.new_answer,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "label-free execution-guided N=4 recompilation after a source fact changes",
        "seed": args.seed,
        "eligible_pool_by_dataset": eligible_pool_by_dataset,
        "selection_limit_per_dataset": args.limit_per_dataset,
        "generation_seconds_including_engine_init": generation_seconds,
        "answer_only_generation_seconds": answer_generation_seconds,
        "datasets": {},
        "answer_only_datasets": {},
    }
    for dataset in paths:
        rows = rows_by_dataset[dataset]
        output = args.output_dir / f"recompile-recovery-{dataset}.jsonl"
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        report["datasets"][dataset] = _summary(cases_by_dataset[dataset], rows)
        if args.answer_adapter:
            answer_rows = answer_rows_by_dataset[dataset]
            answer_output = args.output_dir / f"answer-only-recovery-{dataset}.jsonl"
            answer_output.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in answer_rows
                ),
                encoding="utf-8",
            )
            report["answer_only_datasets"][dataset] = _answer_summary(
                cases_by_dataset[dataset], answer_rows
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
