#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from verijoin.infer import _ANSWER_OUTPUT
from verijoin.recompile_recovery import build_recovery_cases
from verijoin.significance import _paired_bootstrap
from verijoin.text import exact_match, token_f1
from verijoin.vm import execute, parse_program


def _load(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--hotpotqa-predictions", type=Path, required=True)
    parser.add_argument("--twowiki-predictions", type=Path, required=True)
    parser.add_argument("--musique-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit-per-dataset", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    prediction_paths = {
        "hotpotqa": args.hotpotqa_predictions,
        "2wiki": args.twowiki_predictions,
        "musique": args.musique_predictions,
    }
    report: dict[str, object] = {
        "scope": (
            "paired recovery comparison on a SHA-256-selected subset of originally "
            "correct, lineage-certified VeriJoin results; not unconditional QA quality"
        ),
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "datasets": {},
    }
    for offset, (dataset, predictions) in enumerate(prediction_paths.items()):
        cases = build_recovery_cases(
            dataset,
            args.raw_root,
            predictions,
            limit=10**9,
            dataset_variant="distractor" if dataset == "hotpotqa" else None,
        )[: args.limit_per_dataset]
        program_rows = _load(args.output_dir / f"recompile-recovery-{dataset}.jsonl")
        answer_rows = _load(args.output_dir / f"answer-only-recovery-{dataset}.jsonl")
        program_f1: list[float] = []
        answer_f1: list[float] = []
        program_em: list[float] = []
        answer_em: list[float] = []
        for case in cases:
            try:
                result = execute(case.example, parse_program(str(program_rows[case.qid]["output"])))
                program_answer = result.answer if result.valid else ""
            except (ValueError, KeyError, TypeError, json.JSONDecodeError, IndexError):
                program_answer = ""
            match = _ANSWER_OUTPUT.search(str(answer_rows[case.qid]["output"]))
            direct_answer = match.group(1).strip() if match is not None else ""
            program_f1.append(token_f1(program_answer, case.example.answers) if program_answer else 0.0)
            answer_f1.append(token_f1(direct_answer, case.example.answers) if direct_answer else 0.0)
            program_em.append(float(exact_match(program_answer, case.example.answers)))
            answer_em.append(float(exact_match(direct_answer, case.example.answers)))
        f1_difference = np.asarray(program_f1) - np.asarray(answer_f1)
        em_difference = np.asarray(program_em) - np.asarray(answer_em)
        report["datasets"][dataset] = {
            "cases": len(cases),
            "verijoin_strict_f1": 100.0 * float(np.mean(program_f1)),
            "answer_only_f1": 100.0 * float(np.mean(answer_f1)),
            "f1_difference": _paired_bootstrap(
                f1_difference, samples=args.samples, seed=args.seed + offset
            ),
            "verijoin_strict_em": 100.0 * float(np.mean(program_em)),
            "answer_only_em": 100.0 * float(np.mean(answer_em)),
            "em_difference": _paired_bootstrap(
                em_difference, samples=args.samples, seed=args.seed + 100 + offset
            ),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
