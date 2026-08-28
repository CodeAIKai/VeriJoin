from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import iter_examples
from .ranker import RankCandidate, _holdout, candidate_records
from .schema import Example
from .text import exact_match, normalize_answer

DATASETS = ("hotpotqa", "2wiki", "musique")
FEATURE_NAMES = (
    "vote_fraction",
    "best_tier_vote_fraction",
    "best_tier",
    "strict_fraction_all",
    "strict_fraction_group",
    "max_logprob",
    "mean_logprob",
    "max_ranker_score",
    "mean_ranker_score",
    "max_evidence_count",
    "mean_generated_tokens_log1p",
    "answer_words_log1p",
    "answer_in_question",
    "dataset_hotpotqa",
    "dataset_2wiki",
    "dataset_musique",
)


@dataclass(frozen=True, slots=True)
class MetaConfig:
    raw_root: str
    candidates_dir: str
    output: str
    holdout_fraction: float = 0.1
    hidden_size: int = 16
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    epochs: int = 500
    patience: int = 60
    seed: int = 20260825

    @classmethod
    def from_json(cls, path: Path) -> MetaConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class AnswerGroup:
    answer: str
    normalized_answer: str
    records: tuple[RankCandidate, ...]
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GroupExample:
    dataset: str
    qid: str
    groups: tuple[AnswerGroup, ...]
    positive: tuple[bool, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranker_score(row: dict[str, Any], record: RankCandidate) -> float:
    stored = row.get("candidates", [])
    if not isinstance(stored, list) or record.index >= len(stored):
        raise ValueError("candidate index is absent from the scored row")
    candidate = stored[record.index]
    if not isinstance(candidate, dict) or "ranker_score" not in candidate:
        raise ValueError("every answer-valid candidate must have a frozen ranker_score")
    score = float(candidate["ranker_score"])
    if not math.isfinite(score):
        raise ValueError("ranker_score must be finite")
    return score


def answer_groups(example: Example, row: dict[str, Any]) -> tuple[AnswerGroup, ...]:
    """Build deployable answer-group features without reading any supervision field."""
    records = candidate_records(example, row)
    if not records:
        return ()
    grouped: dict[str, list[RankCandidate]] = defaultdict(list)
    for record in records:
        grouped[normalize_answer(record.answer)].append(record)
    candidate_count = len(row.get("candidates", []))
    global_best_tier = max(record.tier for record in records)
    global_best_count = sum(record.tier == global_best_tier for record in records)
    global_strict_count = sum(record.tier == 2 for record in records)
    question = normalize_answer(example.question)
    result: list[AnswerGroup] = []
    for normalized, members in grouped.items():
        logprobs = [record.average_logprob for record in members]
        ranker_scores = [_ranker_score(row, record) for record in members]
        evidence_counts = [
            sum(line.startswith("[") for line in record.passage.splitlines())
            for record in members
        ]
        stored = row.get("candidates", [])
        generated = [
            int(stored[record.index].get("generated_tokens", 0))  # type: ignore[index,union-attr]
            for record in members
        ]
        group_strict = sum(record.tier == 2 for record in members)
        best_tier_votes = sum(record.tier == global_best_tier for record in members)
        dataset_flags = tuple(float(example.dataset == dataset) for dataset in DATASETS)
        features = (
            len(members) / max(1, candidate_count),
            best_tier_votes / max(1, global_best_count),
            float(max(record.tier for record in members)),
            global_strict_count / max(1, len(records)),
            group_strict / len(members),
            max(logprobs),
            sum(logprobs) / len(logprobs),
            max(ranker_scores),
            sum(ranker_scores) / len(ranker_scores),
            float(max(evidence_counts)),
            sum(math.log1p(value) for value in generated) / len(generated),
            math.log1p(len(normalized.split())),
            float(bool(normalized) and normalized in question),
            *dataset_flags,
        )
        result.append(
            AnswerGroup(members[0].answer, normalized, tuple(members), tuple(features))
        )
    return tuple(result)


def _load_training_groups(config: MetaConfig) -> tuple[list[GroupExample], list[GroupExample]]:
    train: list[GroupExample] = []
    holdout: list[GroupExample] = []
    raw_root = Path(config.raw_root)
    candidates_dir = Path(config.candidates_dir)
    for dataset in DATASETS:
        path = candidates_dir / f"stage3-{dataset}-train-n4-scored.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_id = {str(row["id"]): row for row in rows}
        for example in iter_examples(dataset, raw_root, "train"):
            row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
            if row is None:
                continue
            groups = answer_groups(example, row)
            positive = tuple(exact_match(group.answer, example.answers) for group in groups)
            if not any(positive) or all(positive):
                continue
            item = GroupExample(dataset, example.qid, groups, positive)
            target = (
                holdout
                if _holdout(dataset, example.qid, config.holdout_fraction)
                else train
            )
            target.append(item)
    if not train or not holdout:
        raise ValueError("listwise train and holdout groups must both be non-empty")
    return train, holdout


def _normalization(rows: list[GroupExample]) -> tuple[list[float], list[float]]:
    matrix = [group.features for row in rows for group in row.groups]
    means = [sum(row[index] for row in matrix) / len(matrix) for index in range(len(FEATURE_NAMES))]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in matrix) / len(matrix)
        scales.append(max(math.sqrt(variance), 1e-6))
    return means, scales


def _execution_choice(row: GroupExample) -> int:
    best_tier = max(record.tier for group in row.groups for record in group.records)
    return max(
        range(len(row.groups)),
        key=lambda index: (
            sum(record.tier == best_tier for record in row.groups[index].records),
            max(
                (
                    record.average_logprob
                    for record in row.groups[index].records
                    if record.tier == best_tier
                ),
                default=float("-inf"),
            ),
            row.groups[index].normalized_answer,
        ),
    )


def _ranker_choice(row: GroupExample) -> int:
    return max(
        range(len(row.groups)),
        key=lambda index: (
            row.groups[index].features[7],
            len(row.groups[index].records),
            row.groups[index].normalized_answer,
        ),
    )


def _baseline_accuracy(rows: list[GroupExample], kind: str) -> float:
    chooser = _execution_choice if kind == "execution" else _ranker_choice
    return sum(row.positive[chooser(row)] for row in rows) / len(rows)


def _checkpoint_score(features: tuple[float, ...], checkpoint: dict[str, Any]) -> float:
    normalized = [
        (value - mean) / scale
        for value, mean, scale in zip(features, checkpoint["means"], checkpoint["scales"])
    ]
    hidden = [
        math.tanh(sum(weight * value for weight, value in zip(weights, normalized)) + bias)
        for weights, bias in zip(checkpoint["weight1"], checkpoint["bias1"])
    ]
    return sum(weight * value for weight, value in zip(checkpoint["weight2"], hidden)) + float(
        checkpoint["bias2"]
    )


def _checkpoint_accuracy(rows: list[GroupExample], checkpoint: dict[str, Any]) -> float:
    correct = 0
    for row in rows:
        selected = max(
            range(len(row.groups)),
            key=lambda index: (
                _checkpoint_score(row.groups[index].features, checkpoint),
                row.groups[index].normalized_answer,
            ),
        )
        correct += row.positive[selected]
    return correct / len(rows)


def train_meta_ranker(config: MetaConfig) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(config.seed)
    train, holdout = _load_training_groups(config)
    means, scales = _normalization(train)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = nn.Linear(len(FEATURE_NAMES), config.hidden_size)
            self.output = nn.Linear(config.hidden_size, 1)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.output(torch.tanh(self.hidden(values))).view(-1)

    def tensor(group: AnswerGroup) -> torch.Tensor:
        return torch.tensor(
            [(value - mean) / scale for value, mean, scale in zip(group.features, means, scales)],
            dtype=torch.float32,
        )

    train_tensors = [torch.stack([tensor(group) for group in row.groups]) for row in train]
    holdout_tensors = [torch.stack([tensor(group) for group in row.groups]) for row in holdout]
    model = Model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row, values in zip(train, train_tensors):
            scores = model(values)
            positive = torch.tensor(row.positive, dtype=torch.bool)
            losses.append(torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[positive], dim=0))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        model.eval()
        eval_loss = 0.0
        correct = 0
        with torch.inference_mode():
            for row, values in zip(holdout, holdout_tensors):
                scores = model(values)
                positive = torch.tensor(row.positive, dtype=torch.bool)
                eval_loss += float(
                    torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[positive], dim=0)
                )
                correct += row.positive[int(scores.argmax())]
        eval_loss /= len(holdout)
        eval_accuracy = correct / len(holdout)
        if epoch == 1 or epoch % 10 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.detach()),
                    "holdout_loss": eval_loss,
                    "holdout_accuracy": eval_accuracy,
                }
            )
        key = (eval_accuracy, -eval_loss)
        if key > (best_accuracy, -best_loss):
            best_accuracy = eval_accuracy
            best_loss = eval_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("listwise training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint: dict[str, Any] = {
        "version": 1,
        "feature_names": list(FEATURE_NAMES),
        "means": means,
        "scales": scales,
        "weight1": model.hidden.weight.detach().tolist(),
        "bias1": model.hidden.bias.detach().tolist(),
        "weight2": model.output.weight.detach().view(-1).tolist(),
        "bias2": float(model.output.bias.detach()),
        "config": asdict(config),
        "train_questions": len(train),
        "holdout_questions": len(holdout),
        "best_epoch": best_epoch,
        "best_holdout_loss": best_loss,
        "best_holdout_accuracy": best_accuracy,
        "holdout_execution_accuracy": _baseline_accuracy(holdout, "execution"),
        "holdout_ranker_accuracy": _baseline_accuracy(holdout, "ranker"),
        "history": history,
        "uses_gold_labels": "train_only",
        "status": "complete",
    }
    checkpoint["train_accuracy"] = _checkpoint_accuracy(train, checkpoint)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint["output"] = str(output)
    checkpoint["sha256"] = _file_sha256(output)
    return checkpoint


def reselect_with_meta_ranker(
    dataset: str,
    raw_root: Path,
    split: str,
    source: Path,
    output: Path,
    checkpoint_path: Path,
    *,
    dataset_variant: str | None = None,
) -> dict[str, Any]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("meta-ranker feature schema does not match this code version")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["id"]): row for row in rows}
    derived_rows: list[dict[str, Any]] = []
    selected = 0
    for example in iter_examples(dataset, raw_root, split, dataset_variant):
        row = by_id.get(example.qid) or by_id.get(f"{dataset}:{example.qid}")
        if row is None:
            continue
        derived = dict(row)
        stored = [dict(candidate) for candidate in row.get("candidates", [])]
        groups = answer_groups(example, row)
        if groups:
            group_scores = [_checkpoint_score(group.features, checkpoint) for group in groups]
            group_index = max(
                range(len(groups)),
                key=lambda index: (group_scores[index], groups[index].normalized_answer),
            )
            group = groups[group_index]
            for member in group.records:
                stored[member.index]["meta_ranker_score"] = group_scores[group_index]
            record = max(
                group.records,
                key=lambda item: (
                    item.tier,
                    _ranker_score(row, item),
                    item.average_logprob,
                    -item.index,
                ),
            )
            chosen = stored[record.index]
            derived.update(
                {
                    "output": str(chosen["output"]),
                    "finish_reason": str(chosen["finish_reason"]),
                    "generated_tokens": int(chosen["generated_tokens"]),
                    "selection": {
                        "kind": "train_only_listwise_meta_ranker",
                        "selected_index": record.index,
                        "selected_answer_group": group.normalized_answer,
                        "selected_group_score": group_scores[group_index],
                        "answer_groups": len(groups),
                    },
                }
            )
            selected += 1
        derived["candidates"] = stored
        derived_rows.append(derived)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in derived_rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "derived_from": str(source),
        "dataset": dataset,
        "dataset_variant": dataset_variant,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "examples": len(derived_rows),
        "selected": selected,
        "uses_gold_labels": False,
        "status": "complete",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "output": str(output)}
