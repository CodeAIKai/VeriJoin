from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ablation_data import build_ablation_data
from .audit import audit_dataset, audit_dict
from .baseline_eval import evaluate_output_baseline
from .build_data import build_sft
from .build_data import summary_dict as build_summary_dict
from .candidate_eval import analyze_candidates
from .candidate_eval import summary_dict as candidate_summary_dict
from .contract_ablation import evaluate_contract_ablation
from .data import iter_examples
from .evaluate import evaluate_predictions
from .evaluate import summary_dict as evaluation_summary_dict
from .historical_replay import evaluate_hotpot_history
from .infer import InferenceConfig, reselect_predictions, run_inference
from .infer import summary_dict as inference_summary_dict
from .meta_ranker import MetaConfig, reselect_with_meta_ranker, train_meta_ranker
from .operator_analysis import analyze_operator_coverage
from .provenance_baselines import compare_provenance_baselines
from .ranker import build_ranker_pairs, reselect_with_ranker
from .ranker import build_summary_dict as ranker_build_summary_dict
from .ranker import selection_summary_dict as ranker_selection_summary_dict
from .significance import compare_program_predictions, compare_task_predictions
from .stress_eval import evaluate_attacks
from .temporal_wikipedia import (
    collect_title_profiles,
    evaluate_revision_stream,
    fetch_revision_stream,
)
from .update_eval import evaluate_updates
from .update_eval import summary_dict as update_summary_dict
from .update_stress import evaluate_update_stress

DATASETS = ("hotpotqa", "2wiki", "musique")


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verijoin")
    commands = parser.add_subparsers(dest="command", required=True)

    count = commands.add_parser("count", help="count complete official local splits")
    count.add_argument("--raw-root", type=Path, required=True)
    count.add_argument("--split", default="dev")

    audit = commands.add_parser("audit", help="audit gold program compilation coverage")
    audit.add_argument("--raw-root", type=Path, required=True)
    audit.add_argument("--split", default="dev")
    audit.add_argument("--dataset", choices=DATASETS + ("all",), default="all")
    audit.add_argument("--limit", type=int)

    build = commands.add_parser("build-sft", help="compile leakage-safe SFT JSONL")
    build.add_argument("--raw-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--partition", choices=("train", "holdout"), default="train")
    build.add_argument("--holdout-fraction", type=float, default=0.02)
    build.add_argument("--dataset", choices=DATASETS + ("all",), default="all")
    build.add_argument("--strict", action="store_true")
    build.add_argument("--limit-per-dataset", type=int)

    evaluate = commands.add_parser("evaluate", help="strictly execute and score programs")
    evaluate.add_argument("--raw-root", type=Path, required=True)
    evaluate.add_argument("--split", default="dev")
    evaluate.add_argument("--dataset", choices=DATASETS, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--allow-literal", action="store_true")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    evaluate.add_argument("--report", type=Path)

    infer = commands.add_parser("infer", help="run resumable batched local inference")
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--raw-root", type=Path, required=True)
    infer.add_argument("--split", default="dev")
    infer.add_argument("--dataset", choices=DATASETS, required=True)
    infer.add_argument("--output", type=Path, required=True)

    reselect = commands.add_parser(
        "reselect", help="re-select already stored candidates without another model call"
    )
    reselect.add_argument("--raw-root", type=Path, required=True)
    reselect.add_argument("--split", default="dev")
    reselect.add_argument("--dataset", choices=DATASETS, required=True)
    reselect.add_argument("--source", type=Path, required=True)
    reselect.add_argument("--output", type=Path, required=True)
    reselect.add_argument("--selection", choices=("execution", "likelihood"), required=True)
    reselect.add_argument(
        "--task",
        choices=("program", "answer", "citation", "free_literal"),
        default="program",
    )
    reselect.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))

    candidate_eval = commands.add_parser(
        "analyze-candidates",
        help="label-aware oracle/consensus analysis of already stored candidates",
    )
    candidate_eval.add_argument("--raw-root", type=Path, required=True)
    candidate_eval.add_argument("--split", default="dev")
    candidate_eval.add_argument("--dataset", choices=DATASETS, required=True)
    candidate_eval.add_argument("--predictions", type=Path, required=True)
    candidate_eval.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    candidate_eval.add_argument("--report", type=Path)

    build_ranker = commands.add_parser(
        "build-ranker-pairs", help="build train-only positive/negative candidate pairs"
    )
    build_ranker.add_argument("--raw-root", type=Path, required=True)
    build_ranker.add_argument("--candidates-dir", type=Path, required=True)
    build_ranker.add_argument("--train-output", type=Path, required=True)
    build_ranker.add_argument("--eval-output", type=Path, required=True)
    build_ranker.add_argument("--holdout-fraction", type=float, default=0.1)

    train_ranker = commands.add_parser("train-ranker", help="train pairwise BGE LoRA ranker")
    train_ranker.add_argument("--config", type=Path, required=True)

    rank = commands.add_parser(
        "rank-candidates", help="re-select stored candidates with a train-only semantic ranker"
    )
    rank.add_argument("--raw-root", type=Path, required=True)
    rank.add_argument("--split", default="dev")
    rank.add_argument("--dataset", choices=DATASETS, required=True)
    rank.add_argument("--source", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--model", required=True)
    rank.add_argument("--adapter")
    rank.add_argument("--batch-size", type=int, default=64)
    rank.add_argument("--max-length", type=int, default=512)
    rank.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))

    train_meta = commands.add_parser(
        "train-meta-ranker", help="train a train-only listwise answer-group verifier"
    )
    train_meta.add_argument("--config", type=Path, required=True)

    meta = commands.add_parser(
        "meta-rank-candidates", help="re-select scored candidates with the listwise verifier"
    )
    meta.add_argument("--raw-root", type=Path, required=True)
    meta.add_argument("--split", default="dev")
    meta.add_argument("--dataset", choices=DATASETS, required=True)
    meta.add_argument("--source", type=Path, required=True)
    meta.add_argument("--output", type=Path, required=True)
    meta.add_argument("--checkpoint", type=Path, required=True)
    meta.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))

    updates = commands.add_parser(
        "evaluate-updates", help="measure source-version invalidation and selective recomputation"
    )
    updates.add_argument("--raw-root", type=Path, required=True)
    updates.add_argument("--split", default="dev")
    updates.add_argument("--dataset", choices=DATASETS, required=True)
    updates.add_argument("--predictions", type=Path, required=True)
    updates.add_argument("--limit", type=int)
    updates.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    updates.add_argument("--report", type=Path)

    update_stress = commands.add_parser(
        "evaluate-update-stress",
        help="evaluate actual certified programs under five counterfactual update classes",
    )
    update_stress.add_argument("--raw-root", type=Path, required=True)
    update_stress.add_argument("--split", default="dev")
    update_stress.add_argument("--dataset", choices=DATASETS, required=True)
    update_stress.add_argument("--predictions", type=Path, required=True)
    update_stress.add_argument("--limit", type=int)
    update_stress.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    update_stress.add_argument("--report", type=Path)

    ablation = commands.add_parser(
        "evaluate-contracts", help="ablate answer, citation, join, and literal contracts"
    )
    ablation.add_argument("--raw-root", type=Path, required=True)
    ablation.add_argument("--split", default="dev")
    ablation.add_argument("--dataset", choices=DATASETS, required=True)
    ablation.add_argument("--predictions", type=Path, required=True)
    ablation.add_argument("--limit", type=int)
    ablation.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    ablation.add_argument("--report", type=Path)

    operators = commands.add_parser(
        "analyze-operators", help="break accuracy and validity down by operator and join family"
    )
    operators.add_argument("--raw-root", type=Path, required=True)
    operators.add_argument("--split", default="dev")
    operators.add_argument("--dataset", choices=DATASETS, required=True)
    operators.add_argument("--predictions", type=Path, required=True)
    operators.add_argument("--allow-literal", action="store_true")
    operators.add_argument("--limit", type=int)
    operators.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    operators.add_argument("--report", type=Path)

    significance = commands.add_parser(
        "compare-predictions", help="paired-bootstrap two stored program prediction runs"
    )
    significance.add_argument("--raw-root", type=Path, required=True)
    significance.add_argument("--split", default="dev")
    significance.add_argument("--dataset", choices=DATASETS, required=True)
    significance.add_argument("--baseline", type=Path, required=True)
    significance.add_argument("--candidate", type=Path, required=True)
    significance.add_argument("--samples", type=int, default=5000)
    significance.add_argument("--seed", type=int, default=20260825)
    significance.add_argument("--limit", type=int)
    significance.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    significance.add_argument("--report", type=Path)

    task_significance = commands.add_parser(
        "compare-task-predictions",
        help="paired-bootstrap stored program, answer, or citation outputs",
    )
    task_significance.add_argument("--raw-root", type=Path, required=True)
    task_significance.add_argument("--split", default="dev")
    task_significance.add_argument("--dataset", choices=DATASETS, required=True)
    task_significance.add_argument("--baseline", type=Path, required=True)
    task_significance.add_argument("--candidate", type=Path, required=True)
    task_choices = (
        "program-answer",
        "program-strict",
        "program-certified",
        "program-literal",
        "answer",
        "citation",
    )
    task_significance.add_argument("--baseline-task", choices=task_choices, required=True)
    task_significance.add_argument("--candidate-task", choices=task_choices, required=True)
    task_significance.add_argument(
        "--subset", choices=("all", "candidate-eligible"), default="all"
    )
    task_significance.add_argument("--samples", type=int, default=5000)
    task_significance.add_argument("--seed", type=int, default=20260825)
    task_significance.add_argument("--limit", type=int)
    task_significance.add_argument(
        "--dataset-variant", choices=("distractor", "fullwiki")
    )
    task_significance.add_argument("--report", type=Path)

    attacks = commands.add_parser(
        "evaluate-attacks", help="stress lineage certificates with four attack families"
    )
    attacks.add_argument("--raw-root", type=Path, required=True)
    attacks.add_argument("--split", default="dev")
    attacks.add_argument("--dataset", choices=DATASETS, required=True)
    attacks.add_argument("--predictions", type=Path, required=True)
    attacks.add_argument("--limit", type=int)
    attacks.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    attacks.add_argument("--report", type=Path)

    provenance = commands.add_parser(
        "compare-provenance", help="compare provenance size, replay, latency, and invalidation"
    )
    provenance.add_argument("--raw-root", type=Path, required=True)
    provenance.add_argument("--split", default="dev")
    provenance.add_argument("--dataset", choices=DATASETS, required=True)
    provenance.add_argument("--predictions", type=Path, required=True)
    provenance.add_argument("--limit", type=int, default=1000)
    provenance.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    provenance.add_argument("--report", type=Path)

    temporal = commands.add_parser(
        "evaluate-wikipedia-updates", help="evaluate real MediaWiki revision streams"
    )
    temporal.add_argument("--raw-root", type=Path, required=True)
    temporal.add_argument("--split", default="dev")
    temporal.add_argument("--hotpotqa-predictions", type=Path, required=True)
    temporal.add_argument("--twowiki-predictions", type=Path, required=True)
    temporal.add_argument("--musique-predictions", type=Path, required=True)
    temporal.add_argument("--titles-per-dataset", type=int, default=50)
    temporal.add_argument("--revisions", type=int, default=5)
    temporal.add_argument("--batch-size", type=int, default=5)
    temporal.add_argument("--cache", type=Path, required=True)
    temporal.add_argument("--report", type=Path)

    history = commands.add_parser(
        "evaluate-hotpot-history",
        help="evaluate actual certified HotpotQA programs on snapshot-era revisions",
    )
    history.add_argument("--raw-root", type=Path, required=True)
    history.add_argument("--predictions", type=Path, required=True)
    history.add_argument("--history", type=Path, required=True)
    history.add_argument("--report", type=Path)

    ablation_data = commands.add_parser(
        "build-ablation-data", help="rewrite identical SFT rows for an output-contract baseline"
    )
    ablation_data.add_argument("--input", type=Path, required=True)
    ablation_data.add_argument("--output", type=Path, required=True)
    ablation_data.add_argument(
        "--task", choices=("answer", "citation", "free_literal"), required=True
    )

    baseline_eval = commands.add_parser(
        "evaluate-output-baseline", help="score answer-only or citation-only predictions"
    )
    baseline_eval.add_argument("--raw-root", type=Path, required=True)
    baseline_eval.add_argument("--split", default="dev")
    baseline_eval.add_argument("--dataset", choices=DATASETS, required=True)
    baseline_eval.add_argument("--predictions", type=Path, required=True)
    baseline_eval.add_argument("--task", choices=("answer", "citation"), required=True)
    baseline_eval.add_argument("--limit", type=int)
    baseline_eval.add_argument("--dataset-variant", choices=("distractor", "fullwiki"))
    baseline_eval.add_argument("--report", type=Path)

    lengths = commands.add_parser(
        "filter-lengths", help="drop overlength SFT rows without truncating assistant labels"
    )
    lengths.add_argument("--model", required=True)
    lengths.add_argument("--input", type=Path, required=True)
    lengths.add_argument("--output", type=Path, required=True)
    lengths.add_argument("--max-length", type=int, required=True)
    lengths.add_argument("--batch-size", type=int, default=128)

    train = commands.add_parser("train", help="train the QLoRA program compiler")
    train.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "count":
        _json(
            {
                dataset: sum(1 for _ in iter_examples(dataset, args.raw_root, args.split))
                for dataset in DATASETS
            }
        )
        return
    datasets = DATASETS if getattr(args, "dataset", "all") == "all" else (args.dataset,)
    if args.command == "audit":
        _json(
            [
                audit_dict(audit_dataset(dataset, args.raw_root, args.split, args.limit))
                for dataset in datasets
            ]
        )
        return
    if args.command == "build-sft":
        summary = build_sft(
            datasets,
            args.raw_root,
            args.output,
            partition=args.partition,
            holdout_fraction=args.holdout_fraction,
            require_strict=args.strict,
            limit_per_dataset=args.limit_per_dataset,
        )
        _json(build_summary_dict(summary))
        return
    if args.command == "evaluate":
        summary = evaluate_predictions(
            args.dataset,
            args.raw_root,
            args.split,
            args.predictions,
            allow_literal=args.allow_literal,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
        )
        payload = evaluation_summary_dict(summary)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "infer":
        config = InferenceConfig.from_json(args.config)
        summary = run_inference(config, args.dataset, args.raw_root, args.split, args.output)
        _json(inference_summary_dict(summary))
        return
    if args.command == "reselect":
        count = reselect_predictions(
            args.dataset,
            args.raw_root,
            args.split,
            args.source,
            args.output,
            execution_guided=args.selection == "execution",
            task=args.task,
            dataset_variant=args.dataset_variant,
        )
        _json({"examples": count, "output": str(args.output), "selection": args.selection})
        return
    if args.command == "analyze-candidates":
        summary = analyze_candidates(
            args.dataset,
            args.raw_root,
            args.split,
            args.predictions,
            dataset_variant=args.dataset_variant,
        )
        payload = candidate_summary_dict(summary)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "build-ranker-pairs":
        summary = build_ranker_pairs(
            DATASETS,
            args.raw_root,
            args.candidates_dir,
            args.train_output,
            args.eval_output,
            holdout_fraction=args.holdout_fraction,
        )
        _json(ranker_build_summary_dict(summary))
        return
    if args.command == "train-ranker":
        from .ranker_train import RankerConfig
        from .ranker_train import train_ranker as run_ranker_training

        _json(run_ranker_training(RankerConfig.from_json(args.config)))
        return
    if args.command == "rank-candidates":
        summary = reselect_with_ranker(
            args.dataset,
            args.raw_root,
            args.split,
            args.source,
            args.output,
            args.model,
            args.adapter,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dataset_variant=args.dataset_variant,
        )
        _json(ranker_selection_summary_dict(summary))
        return
    if args.command == "train-meta-ranker":
        _json(train_meta_ranker(MetaConfig.from_json(args.config)))
        return
    if args.command == "meta-rank-candidates":
        _json(
            reselect_with_meta_ranker(
                args.dataset,
                args.raw_root,
                args.split,
                args.source,
                args.output,
                args.checkpoint,
                dataset_variant=args.dataset_variant,
            )
        )
        return
    if args.command == "evaluate-updates":
        summary = evaluate_updates(
            args.dataset,
            args.raw_root,
            args.split,
            args.predictions,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
        )
        payload = update_summary_dict(summary)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "evaluate-update-stress":
        payload = evaluate_update_stress(
            args.dataset,
            args.raw_root,
            args.split,
            args.predictions,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "evaluate-contracts":
        payload = evaluate_contract_ablation(
            args.dataset,
            args.raw_root,
            args.split,
            args.predictions,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "analyze-operators":
        payload = analyze_operator_coverage(
            args.dataset,
            args.raw_root,
            args.split,
            args.predictions,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
            allow_literal=args.allow_literal,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "compare-predictions":
        payload = compare_program_predictions(
            args.dataset,
            args.raw_root,
            args.split,
            args.baseline,
            args.candidate,
            samples=args.samples,
            seed=args.seed,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "compare-task-predictions":
        payload = compare_task_predictions(
            args.dataset,
            args.raw_root,
            args.split,
            args.baseline,
            args.candidate,
            baseline_task=args.baseline_task,
            candidate_task=args.candidate_task,
            subset=args.subset,
            samples=args.samples,
            seed=args.seed,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "evaluate-attacks":
        payload = evaluate_attacks(
            args.dataset,
            args.raw_root,
            args.split,
            args.predictions,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "compare-provenance":
        payload = compare_provenance_baselines(
            args.dataset,
            args.raw_root,
            args.split,
            args.predictions,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "evaluate-wikipedia-updates":
        prediction_paths = {
            "hotpotqa": args.hotpotqa_predictions,
            "2wiki": args.twowiki_predictions,
            "musique": args.musique_predictions,
        }
        profiles = []
        for dataset, path in prediction_paths.items():
            profiles.extend(
                collect_title_profiles(
                    dataset,
                    args.raw_root,
                    args.split,
                    path,
                    count=args.titles_per_dataset,
                )
            )
        stream = fetch_revision_stream(
            profiles,
            args.cache,
            revisions=args.revisions,
            batch_size=args.batch_size,
        )
        payload = evaluate_revision_stream(stream)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "evaluate-hotpot-history":
        payload = evaluate_hotpot_history(
            args.raw_root, args.predictions, args.history
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "build-ablation-data":
        _json(build_ablation_data(args.input, args.output, args.task))
        return
    if args.command == "evaluate-output-baseline":
        payload = evaluate_output_baseline(
            args.dataset,
            args.raw_root,
            args.split,
            args.predictions,
            task=args.task,
            limit=args.limit,
            dataset_variant=args.dataset_variant,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _json(payload)
        return
    if args.command == "filter-lengths":
        from dataclasses import asdict

        from transformers import AutoTokenizer

        from .sft import filter_sft_by_length

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        summary = filter_sft_by_length(
            args.input, args.output, tokenizer, args.max_length, args.batch_size
        )
        _json(asdict(summary))
        return
    if args.command == "train":
        from .sft import SFTConfig, train

        train(SFTConfig.from_json(args.config))
        return


if __name__ == "__main__":
    main()
