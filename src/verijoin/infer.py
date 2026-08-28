from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from transformers import AutoTokenizer

from .data import iter_examples
from .prompt import inference_messages
from .schema import Example
from .text import normalize_answer
from .vm import execute, parse_program


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    model: str
    adapter: str | None
    backend: Literal["vllm", "transformers"] = "vllm"
    max_model_len: int = 6656
    max_input_tokens: int = 6144
    max_new_tokens: int = 384
    chunk_size: int = 256
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    seed: int = 20260825
    max_examples: int | None = None
    hf_batch_size: int = 1
    dataset_variant: Literal["distractor", "fullwiki"] | None = None
    num_candidates: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    execution_guided: bool = True
    skip_overlength: bool = False
    task: Literal["program", "answer", "citation", "free_literal"] = "program"

    @classmethod
    def from_json(cls, path: Path) -> InferenceConfig:
        config = cls(**json.loads(path.read_text(encoding="utf-8")))
        if config.num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if config.num_candidates > 1 and config.temperature <= 0:
            raise ValueError("multi-candidate decoding requires temperature > 0")
        if not 0 < config.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if config.max_input_tokens + config.max_new_tokens > config.max_model_len:
            raise ValueError("input and output token budgets exceed max_model_len")
        return config


@dataclass(frozen=True, slots=True)
class InferenceSummary:
    dataset: str
    dataset_variant: str
    split: str
    output: str
    total_examples: int
    previously_complete: int
    newly_written: int
    new_input_tokens: int
    new_output_tokens: int
    render_seconds: float
    engine_init_seconds: float
    generation_seconds: float
    examples_per_second: float
    output_tokens_per_second: float
    skipped_overlength: int
    skipped_overlength_ids: tuple[str, ...]


def _chunks(values: list[tuple[Example, str]], size: int) -> Iterable[list[tuple[Example, str]]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                result.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                # A partially written final line is ignored and repaired on resume.
                continue
    return result


def _render_pending(
    dataset: str,
    raw_root: Path,
    split: str,
    tokenizer: Any,
    complete: set[str],
    max_input_tokens: int,
    max_examples: int | None,
    dataset_variant: str | None,
    skip_overlength: bool,
    task: str,
) -> tuple[int, list[tuple[Example, str]], int, tuple[str, ...]]:
    total = 0
    input_tokens = 0
    pending: list[tuple[Example, str]] = []
    skipped: list[str] = []
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        if max_examples is not None and total >= max_examples:
            break
        total += 1
        if example.qid in complete or f"{dataset}:{example.qid}" in complete:
            continue
        prompt = tokenizer.apply_chat_template(
            inference_messages(example, task), tokenize=False, add_generation_prompt=True
        )
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if length > max_input_tokens:
            if skip_overlength:
                skipped.append(f"{example.qid}:{length}")
                continue
            raise ValueError(
                f"{dataset}/{example.qid} has {length} input tokens, above the declared "
                f"no-truncation limit {max_input_tokens}"
            )
        pending.append((example, prompt))
        input_tokens += length
    return total, pending, input_tokens, tuple(skipped)


def _select_candidate(
    example: Example,
    candidates: list[tuple[str, float]],
    *,
    execution_guided: bool,
) -> tuple[int, dict[str, object]]:
    """Select without labels: executable tier, answer consensus, then model likelihood."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    evaluated: list[tuple[int, int, str, float, bool]] = []
    for index, (candidate, average_logprob) in enumerate(candidates):
        tier = 0
        answer = ""
        lineage_certified = False
        if execution_guided:
            try:
                result = execute(example, parse_program(candidate))
                lineage_certified = result.lineage_certified
                if result.valid:
                    tier = 2
                    answer = normalize_answer(result.answer)
                elif result.answer_valid:
                    tier = 1
                    answer = normalize_answer(result.candidate_answer)
            except (ValueError, IndexError, KeyError, TypeError, json.JSONDecodeError):
                pass
        evaluated.append((index, tier, answer, average_logprob, lineage_certified))

    if not execution_guided:
        selected = max(evaluated, key=lambda item: (item[3], -item[0]))
    else:
        best_tier = max(item[1] for item in evaluated)
        eligible = [item for item in evaluated if item[1] == best_tier]
        answer_counts = Counter(item[2] for item in eligible if item[2])
        if answer_counts:
            best_answer = max(
                answer_counts,
                key=lambda answer: (
                    answer_counts[answer],
                    max(item[3] for item in eligible if item[2] == answer),
                    answer,
                ),
            )
            eligible = [item for item in eligible if item[2] == best_answer]
        selected = max(eligible, key=lambda item: (item[3], -item[0]))
    return selected[0], {
        "selected_index": selected[0],
        "lineage_certified_candidates": sum(item[4] for item in evaluated),
        "strict_valid_candidates": sum(item[1] >= 2 for item in evaluated),
        "answer_valid_candidates": sum(item[1] >= 1 for item in evaluated),
        "selected_tier": selected[1],
        "selected_answer_consensus": sum(
            item[1] == selected[1] and bool(selected[2]) and item[2] == selected[2]
            for item in evaluated
        ),
    }


def _average_logprob(cumulative_logprob: float | None, token_count: int) -> float:
    """Return length-normalized model likelihood and reject unavailable scores."""
    if cumulative_logprob is None:
        raise RuntimeError(
            "vLLM did not return cumulative log probability; SamplingParams.logprobs "
            "must be a non-None value"
        )
    return float(cumulative_logprob) / max(1, token_count)


def run_vllm(
    config: InferenceConfig,
    dataset: str,
    raw_root: Path,
    split: str,
    output: Path,
) -> InferenceSummary:
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as error:
        raise RuntimeError("the selected vLLM backend is not installed") from error

    render_start = time.perf_counter()
    tokenizer_source = config.adapter or config.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    complete = _existing_ids(output)
    total, pending, input_tokens, skipped = _render_pending(
        dataset,
        raw_root,
        split,
        tokenizer,
        complete,
        config.max_input_tokens,
        config.max_examples,
        config.dataset_variant,
        config.skip_overlength,
        config.task,
    )
    render_seconds = time.perf_counter() - render_start
    if not pending:
        return InferenceSummary(
            dataset,
            config.dataset_variant or ("distractor" if dataset == "hotpotqa" else "default"),
            split,
            str(output),
            total,
            len(complete),
            0,
            0,
            0,
            render_seconds,
            0.0,
            0.0,
            len(skipped),
            skipped,
            0.0,
            0.0,
        )

    engine_start = time.perf_counter()
    engine = LLM(
        model=config.model,
        tokenizer=config.model,
        trust_remote_code=True,
        tensor_parallel_size=config.tensor_parallel_size,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_lora=config.adapter is not None,
        max_lora_rank=64,
        seed=config.seed,
    )
    engine_init_seconds = time.perf_counter() - engine_start
    sampling = SamplingParams(
        n=config.num_candidates,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        seed=config.seed,
        # Zero returns the sampled token's probability without requesting any
        # additional top-k alternatives.  A non-None value is required for
        # CompletionOutput.cumulative_logprob to be populated by vLLM.
        logprobs=0,
    )
    lora_request = LoRARequest("verijoin", 1, config.adapter) if config.adapter else None
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    output_tokens = 0
    generation_start = time.perf_counter()
    with output.open("a", encoding="utf-8") as handle:
        for chunk in _chunks(pending, config.chunk_size):
            generated = engine.generate(
                [prompt for _, prompt in chunk],
                sampling_params=sampling,
                lora_request=lora_request,
                use_tqdm=True,
            )
            for (example, _), result in zip(chunk, generated):
                candidates = [
                    (
                        candidate.text,
                        _average_logprob(candidate.cumulative_logprob, len(candidate.token_ids)),
                    )
                    for candidate in result.outputs
                ]
                selected_index, selection = _select_candidate(
                    example, candidates, execution_guided=config.execution_guided
                )
                selected = result.outputs[selected_index]
                row = {
                    "id": example.qid,
                    "dataset": dataset,
                    "output": selected.text,
                    "finish_reason": str(selected.finish_reason),
                    "generated_tokens": len(selected.token_ids),
                    "selection": selection,
                }
                if config.num_candidates > 1:
                    row["candidates"] = [
                        {
                            "output": candidate.text,
                            "finish_reason": str(candidate.finish_reason),
                            "generated_tokens": len(candidate.token_ids),
                            "average_logprob": candidates[index][1],
                        }
                        for index, candidate in enumerate(result.outputs)
                    ]
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                written += 1
                output_tokens += sum(len(candidate.token_ids) for candidate in result.outputs)
    generation_seconds = time.perf_counter() - generation_start
    return InferenceSummary(
        dataset,
        config.dataset_variant or ("distractor" if dataset == "hotpotqa" else "default"),
        split,
        str(output),
        total,
        len(complete),
        written,
        input_tokens,
        output_tokens,
        render_seconds,
        engine_init_seconds,
        generation_seconds,
        written / generation_seconds if generation_seconds else 0.0,
        output_tokens / generation_seconds if generation_seconds else 0.0,
        len(skipped),
        skipped,
    )


def run_transformers(
    config: InferenceConfig,
    dataset: str,
    raw_root: Path,
    split: str,
    output: Path,
) -> InferenceSummary:
    if config.num_candidates != 1:
        raise ValueError("multi-candidate execution-guided decoding currently requires vLLM")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    render_start = time.perf_counter()
    tokenizer_source = config.adapter or config.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    complete = _existing_ids(output)
    total, pending, input_tokens, skipped = _render_pending(
        dataset,
        raw_root,
        split,
        tokenizer,
        complete,
        config.max_input_tokens,
        config.max_examples,
        config.dataset_variant,
        config.skip_overlength,
        config.task,
    )
    render_seconds = time.perf_counter() - render_start
    if not pending:
        return InferenceSummary(
            dataset,
            config.dataset_variant or ("distractor" if dataset == "hotpotqa" else "default"),
            split,
            str(output),
            total,
            len(complete),
            0,
            0,
            0,
            render_seconds,
            0.0,
            0.0,
            len(skipped),
            skipped,
            0.0,
            0.0,
        )

    engine_start = time.perf_counter()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    if config.adapter:
        model = PeftModel.from_pretrained(model, config.adapter)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    engine_init_seconds = time.perf_counter() - engine_start
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    output_tokens = 0
    generation_start = time.perf_counter()
    with output.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for chunk in _chunks(pending, config.hf_batch_size):
            encoded = tokenizer(
                [prompt for _, prompt in chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to("cuda")
            sequences = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            prompt_width = encoded["input_ids"].shape[1]
            for (qid, _), sequence in zip(chunk, sequences):
                token_ids = sequence[prompt_width:].tolist()
                if tokenizer.eos_token_id in token_ids:
                    stop = token_ids.index(tokenizer.eos_token_id) + 1
                    token_ids = token_ids[:stop]
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                row = {
                    "id": qid,
                    "dataset": dataset,
                    "output": text,
                    "finish_reason": "length"
                    if len(token_ids) >= config.max_new_tokens
                    else "stop",
                    "generated_tokens": len(token_ids),
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                written += 1
                output_tokens += len(token_ids)
    generation_seconds = time.perf_counter() - generation_start
    return InferenceSummary(
        dataset,
        config.dataset_variant or ("distractor" if dataset == "hotpotqa" else "default"),
        split,
        str(output),
        total,
        len(complete),
        written,
        input_tokens,
        output_tokens,
        render_seconds,
        engine_init_seconds,
        generation_seconds,
        written / generation_seconds if generation_seconds else 0.0,
        output_tokens / generation_seconds if generation_seconds else 0.0,
        len(skipped),
        skipped,
    )


def run_inference(
    config: InferenceConfig,
    dataset: str,
    raw_root: Path,
    split: str,
    output: Path,
) -> InferenceSummary:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    identity = {
        "config": asdict(config),
        "dataset": dataset,
        "split": split,
    }
    if output.exists() and output.stat().st_size:
        if not manifest_path.exists():
            raise ValueError(f"refusing to resume {output} without a provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_identity = {
            "config": previous.get("config"),
            "dataset": previous.get("dataset", previous.get("summary", {}).get("dataset")),
            "split": previous.get("split", previous.get("summary", {}).get("split")),
        }
        if previous_identity != identity:
            raise ValueError(f"refusing to mix an incompatible inference run into {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({**identity, "status": "running"}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if config.backend == "transformers":
        summary = run_transformers(config, dataset, raw_root, split, output)
    else:
        summary = run_vllm(config, dataset, raw_root, split, output)
    manifest = {
        **identity,
        "status": "complete",
        "summary": asdict(summary),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def reselect_predictions(
    dataset: str,
    raw_root: Path,
    split: str,
    source: Path,
    output: Path,
    *,
    execution_guided: bool,
    dataset_variant: str | None = None,
) -> int:
    """Re-select stored candidates without another model call or access to supervision."""
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    selected_rows: list[dict[str, object]] = []
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        stored = row.get("candidates")
        if not isinstance(stored, list) or not stored:
            raise ValueError(f"{example.qid} has no stored candidates to re-select")
        candidates = [
            (str(candidate["output"]), float(candidate["average_logprob"])) for candidate in stored
        ]
        index, selection = _select_candidate(example, candidates, execution_guided=execution_guided)
        chosen = stored[index]
        derived = dict(row)
        derived.update(
            {
                "output": str(chosen["output"]),
                "finish_reason": str(chosen["finish_reason"]),
                "generated_tokens": int(chosen["generated_tokens"]),
                "selection": selection,
            }
        )
        selected_rows.append(derived)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in selected_rows
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
                "execution_guided": execution_guided,
                "examples": len(selected_rows),
                "additional_model_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(selected_rows)


def summary_dict(summary: InferenceSummary) -> dict[str, object]:
    return asdict(summary)
