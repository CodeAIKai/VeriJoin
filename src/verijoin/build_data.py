from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .compiler import compile_gold
from .data import iter_examples
from .prompt import messages
from .vm import execute


@dataclass(frozen=True, slots=True)
class BuildSummary:
    output: str
    total_seen: int
    written: int
    skipped_unanchored: int
    skipped_invalid: int
    skipped_compile_error: int
    per_dataset: dict[str, int]
    sha256: str


def _stable_bucket(dataset: str, qid: str, modulus: int = 10_000) -> int:
    digest = hashlib.sha256(f"verijoin-v1\n{dataset}\n{qid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sft(
    datasets: Iterable[str],
    raw_root: Path,
    output: Path,
    *,
    split: str = "train",
    holdout_fraction: float = 0.02,
    partition: str = "train",
    require_strict: bool = False,
    limit_per_dataset: int | None = None,
) -> BuildSummary:
    if partition not in {"train", "holdout"}:
        raise ValueError("partition must be train or holdout")
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in [0, 1)")
    cutoff = int(holdout_fraction * 10_000)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    total = written = unanchored = invalid = compile_error = 0
    counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for dataset in datasets:
            seen_dataset = 0
            for example in iter_examples(dataset, raw_root, split):
                if limit_per_dataset is not None and seen_dataset >= limit_per_dataset:
                    break
                seen_dataset += 1
                total += 1
                in_holdout = _stable_bucket(dataset, example.qid) < cutoff
                if (partition == "holdout") != in_holdout:
                    continue
                try:
                    program = compile_gold(example)
                except (ValueError, IndexError, KeyError):
                    compile_error += 1
                    continue
                if program.answer.op == "literal":
                    unanchored += 1
                    continue
                result = execute(example, program)
                if require_strict and not result.valid:
                    invalid += 1
                    continue
                row = {
                    "id": f"{dataset}:{example.qid}",
                    "dataset": dataset,
                    "messages": messages(example, program),
                    "program": program.to_dict(),
                    "strict_gold_valid": result.valid,
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                counts[dataset] += 1
    temporary.replace(output)
    return BuildSummary(
        output=str(output),
        total_seen=total,
        written=written,
        skipped_unanchored=unanchored,
        skipped_invalid=invalid,
        skipped_compile_error=compile_error,
        per_dataset=dict(counts),
        sha256=_sha256(output),
    )


def summary_dict(summary: BuildSummary) -> dict[str, object]:
    return asdict(summary)
