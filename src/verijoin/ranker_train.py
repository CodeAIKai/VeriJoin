from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RankerConfig:
    model: str
    train_file: str
    eval_file: str
    output_dir: str
    max_length: int = 512
    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.05
    seed: int = 20260825

    @classmethod
    def from_json(cls, path: Path) -> RankerConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class PairDataset:
    def __init__(self, path: Path) -> None:
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def _evaluate(model: Any, tokenizer: Any, loader: Any, max_length: int) -> tuple[float, float]:
    import torch
    from torch.nn import functional

    model.eval()
    loss_sum = correct = examples = 0.0
    with torch.inference_mode():
        for rows in loader:
            pairs = [[row["query"], row["positive"]] for row in rows] + [
                [row["query"], row["negative"]] for row in rows
            ]
            encoded = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to("cuda")
            scores = model(**encoded, return_dict=True).logits.view(-1).float()
            count = len(rows)
            margins = scores[:count] - scores[count:]
            loss_sum += float(functional.softplus(-margins).sum())
            correct += float((margins > 0).sum())
            examples += count
    return loss_sum / max(1.0, examples), correct / max(1.0, examples)


def train_ranker(config: RankerConfig) -> dict[str, object]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.nn import functional
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
        set_seed,
    )

    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    train_path = Path(config.train_file)
    eval_path = Path(config.eval_file)
    output_dir = Path(config.output_dir)
    train_data = PairDataset(train_path)
    eval_data = PairDataset(eval_path)
    if not train_data.rows or not eval_data.rows:
        raise ValueError("ranker train and eval files must both be non-empty")
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    base = AutoModelForSequenceClassification.from_pretrained(
        config.model, torch_dtype=torch.bfloat16
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["query", "key", "value"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base, lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = model.cuda()

    generator = torch.Generator().manual_seed(config.seed)
    collate = lambda rows: rows
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=config.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation)
    total_updates = updates_per_epoch * config.epochs
    warmup_steps = int(total_updates * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    update = 0
    train_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_pairs = 0
        for step, rows in enumerate(train_loader, start=1):
            pairs = [[row["query"], row["positive"]] for row in rows] + [
                [row["query"], row["negative"]] for row in rows
            ]
            encoded = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to("cuda")
            scores = model(**encoded, return_dict=True).logits.view(-1).float()
            count = len(rows)
            loss = functional.softplus(-(scores[:count] - scores[count:])).mean()
            (loss / config.gradient_accumulation).backward()
            epoch_loss += float(loss.detach()) * count
            epoch_pairs += count
            if step % config.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
        eval_loss, eval_accuracy = _evaluate(model, tokenizer, eval_loader, config.max_length)
        metrics: dict[str, float | int] = {
            "epoch": epoch,
            "updates": update,
            "train_loss": epoch_loss / max(1, epoch_pairs),
            "eval_loss": eval_loss,
            "eval_pair_accuracy": eval_accuracy,
        }
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)

    elapsed = time.perf_counter() - train_start
    manifest = {
        "config": asdict(config),
        "train_pairs": len(train_data),
        "eval_pairs": len(eval_data),
        "train_sha256": _file_sha256(train_path),
        "eval_sha256": _file_sha256(eval_path),
        "history": history,
        "best_epoch": best_epoch,
        "best_eval_loss": best_loss,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "status": "complete",
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
