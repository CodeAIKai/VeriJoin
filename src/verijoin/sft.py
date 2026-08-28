from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SFTConfig:
    model: str
    train_file: str
    eval_file: str | None
    output_dir: str
    init_adapter: str | None = None
    max_length: int = 6144
    learning_rate: float = 1.5e-4
    epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation: int = 16
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    eval_max_examples: int = 512
    max_steps: int = -1
    balance_datasets: bool = True
    balance_operators: bool = True
    seed: int = 20260825

    @classmethod
    def from_json(cls, path: Path) -> SFTConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class ProgramDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        max_length: int,
        max_examples: int | None = None,
        balance_operators: bool = True,
        subset_seed: int = 20260825,
    ) -> None:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.rows = (
            _balanced_subset(rows, max_examples, subset_seed)
            if max_examples is not None and len(rows) > max_examples
            else rows
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        dataset_counts = Counter(str(row["dataset"]) for row in self.rows)
        strata = Counter(
            (str(row["dataset"]), str(row["program"]["answer"]["op"])) for row in self.rows
        )
        self.sample_weights = []
        for row in self.rows:
            dataset = str(row["dataset"])
            operator = str(row["program"]["answer"]["op"])
            dataset_weight = 1.0 / dataset_counts[dataset]
            operator_boost = (
                math.sqrt(dataset_counts[dataset] / strata[(dataset, operator)])
                if balance_operators
                else 1.0
            )
            self.sample_weights.append(dataset_weight * operator_boost)
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        all_messages = row["messages"]
        prompt_messages = all_messages[:-1]
        full_text = self.tokenizer.apply_chat_template(
            all_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(full_text, truncation=False, add_special_tokens=False)
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
        input_ids = encoded["input_ids"]
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"{row.get('id', index)} has {len(input_ids)} tokens, above max_length="
                f"{self.max_length}; run `verijoin filter-lengths` instead of truncating labels"
            )
        labels = list(input_ids)
        prefix = min(len(prompt_ids), len(labels))
        if prefix >= len(labels):
            raise ValueError(f"{row.get('id', index)} contains no supervised assistant tokens")
        labels[:prefix] = [IGNORE_INDEX] * prefix
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _balanced_subset(
    rows: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministically balance a small validation subset by dataset, then answer operator."""
    if limit <= 0:
        return []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[str(row["dataset"])][str(row["program"]["answer"]["op"])].append(row)
    for operators in grouped.values():
        for values in operators.values():
            values.sort(
                key=lambda row: hashlib.sha256(
                    f"{seed}:{row.get('id', '')}".encode()
                ).digest()
            )

    datasets = sorted(grouped)
    quotas = {
        dataset: limit // len(datasets) + int(index < limit % len(datasets))
        for index, dataset in enumerate(datasets)
    }
    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        operators = grouped[dataset]
        names = sorted(operators)
        offsets = Counter({name: 0 for name in names})
        while sum(offsets.values()) < quotas[dataset]:
            progressed = False
            for name in names:
                offset = offsets[name]
                if offset >= len(operators[name]):
                    continue
                selected.append(operators[name][offset])
                offsets[name] += 1
                progressed = True
                if sum(offsets.values()) >= quotas[dataset]:
                    break
            if not progressed:
                break
    return selected


@dataclass(frozen=True, slots=True)
class LengthFilterSummary:
    input: str
    output: str
    max_length: int
    seen: int
    kept: int
    dropped: int
    counts_seen: dict[str, int]
    counts_kept: dict[str, int]
    min_tokens: int
    p50_tokens: int
    p95_tokens: int
    p99_tokens: int
    max_tokens: int
    longest: tuple[tuple[str, int], ...]
    sha256: str


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[int(fraction * (len(sorted_values) - 1))]


def filter_sft_by_length(
    input_path: Path,
    output_path: Path,
    tokenizer: Any,
    max_length: int,
    batch_size: int = 128,
) -> LengthFilterSummary:
    """Write only examples whose complete chat target fits; never truncate supervision."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts_seen: Counter[str] = Counter()
    counts_kept: Counter[str] = Counter()
    lengths: list[int] = []
    longest: list[tuple[str, int]] = []
    digest = hashlib.sha256()
    seen = kept = 0

    def flush(rows: list[tuple[str, dict[str, Any]]], handle: Any) -> None:
        nonlocal seen, kept
        if not rows:
            return
        rendered = [
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            for _, row in rows
        ]
        tokenized = tokenizer(rendered, add_special_tokens=False, padding=False, truncation=False)[
            "input_ids"
        ]
        for (line, row), ids in zip(rows, tokenized):
            dataset = str(row["dataset"])
            qid = str(row.get("id", seen))
            length = len(ids)
            seen += 1
            counts_seen[dataset] += 1
            lengths.append(length)
            longest.append((qid, length))
            longest.sort(key=lambda item: item[1], reverse=True)
            del longest[10:]
            if length > max_length:
                continue
            encoded = line.encode("utf-8")
            handle.write(line)
            digest.update(encoded)
            kept += 1
            counts_kept[dataset] += 1

    pending: list[tuple[str, dict[str, Any]]] = []
    with (
        input_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            if not line.strip():
                continue
            pending.append((line, json.loads(line)))
            if len(pending) >= batch_size:
                flush(pending, target)
                pending.clear()
        flush(pending, target)
    ordered = sorted(lengths)
    return LengthFilterSummary(
        input=str(input_path),
        output=str(output_path),
        max_length=max_length,
        seen=seen,
        kept=kept,
        dropped=seen - kept,
        counts_seen=dict(sorted(counts_seen.items())),
        counts_kept=dict(sorted(counts_kept.items())),
        min_tokens=ordered[0] if ordered else 0,
        p50_tokens=_percentile(ordered, 0.50),
        p95_tokens=_percentile(ordered, 0.95),
        p99_tokens=_percentile(ordered, 0.99),
        max_tokens=ordered[-1] if ordered else 0,
        longest=tuple(longest),
        sha256=digest.hexdigest(),
    )


class LeftPadCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        result: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feature in features:
            padding = width - len(feature["input_ids"])
            result["input_ids"].append(
                torch.nn.functional.pad(feature["input_ids"], (padding, 0), value=self.pad_token_id)
            )
            result["attention_mask"].append(
                torch.nn.functional.pad(feature["attention_mask"], (padding, 0), value=0)
            )
            result["labels"].append(
                torch.nn.functional.pad(feature["labels"], (padding, 0), value=IGNORE_INDEX)
            )
        return {key: torch.stack(value) for key, value in result.items()}


class BalancedTrainer(Trainer):
    def _get_train_sampler(
        self, train_dataset: Dataset[dict[str, torch.Tensor]] | None = None
    ) -> WeightedRandomSampler | None:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or not hasattr(dataset, "sample_weights"):
            return None
        return WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(self.args.seed),
        )


def train(config: SFTConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = (
        PeftModel.from_pretrained(model, config.init_adapter, is_trainable=True)
        if config.init_adapter
        else get_peft_model(model, lora)
    )
    model.print_trainable_parameters()

    train_dataset = ProgramDataset(
        Path(config.train_file),
        tokenizer,
        config.max_length,
        balance_operators=config.balance_operators,
    )
    eval_dataset = (
        ProgramDataset(
            Path(config.eval_file),
            tokenizer,
            config.max_length,
            config.eval_max_examples,
            balance_operators=False,
            subset_seed=config.seed,
        )
        if config.eval_file
        else None
    )
    arguments = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        optim="paged_adamw_8bit",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer_class = BalancedTrainer if config.balance_datasets else Trainer
    trainer = trainer_class(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=LeftPadCollator(tokenizer.pad_token_id),
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    train_result = trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    eval_history = [
        entry for entry in trainer.state.log_history if "eval_loss" in entry
    ]
    manifest = {
        "config": asdict(config),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "effective_sample_exposures": (
            config.max_steps * config.batch_size * config.gradient_accumulation
            if config.max_steps > 0
            else None
        ),
        "train_file_sha256": _file_sha256(Path(config.train_file)),
        "eval_file_sha256": _file_sha256(Path(config.eval_file)) if config.eval_file else None,
        "init_adapter_sha256": _file_sha256(
            Path(config.init_adapter) / "adapter_model.safetensors"
            if config.init_adapter
            else None
        ),
        "train_metrics": train_result.metrics,
        "eval_history": eval_history,
        "final_eval_metrics": eval_history[-1] if eval_history else None,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
    }
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
