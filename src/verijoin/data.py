from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .schema import DecompositionStep, Document, Example
from .text import clean, split_sentences

DATASET_DIRS = {
    "hotpotqa": "hotpot_qa",
    "2wiki": "2WikiMultihopQA",
    "musique": "MuSiQue",
}

SUPERVISION_KEYS = {
    "answer",
    "answers",
    "answer_aliases",
    "support",
    "supporting_facts",
    "is_supporting",
    "question_decomposition",
    "evidences",
    "gold",
}


def source_files(
    dataset: str,
    root: Path,
    split: str,
    dataset_variant: str | None = None,
) -> list[Path]:
    base = root / DATASET_DIRS[dataset]
    if dataset == "hotpotqa":
        variant = dataset_variant or "distractor"
        if variant not in {"distractor", "fullwiki"}:
            raise ValueError(f"unknown HotpotQA variant: {variant}")
        mapped = "validation" if split in {"dev", "validation"} else split
        return sorted((base / variant).glob(f"{mapped}-*.parquet"))
    if dataset_variant is not None:
        raise ValueError(f"dataset variant is only supported for HotpotQA, not {dataset}")
    if dataset == "2wiki":
        mapped = "dev" if split in {"dev", "validation"} else split
        return sorted(base.glob(f"{mapped}*.parquet"))
    if dataset == "musique":
        mapped = "dev" if split in {"dev", "validation"} else split
        return sorted(base.glob(f"musique_ans_v1.0_{mapped}.jsonl"))
    raise ValueError(f"unknown dataset: {dataset}")


def iter_raw(
    dataset: str,
    root: Path,
    split: str,
    dataset_variant: str | None = None,
) -> Iterator[dict[str, Any]]:
    paths = source_files(dataset, root, split, dataset_variant)
    if not paths:
        raise FileNotFoundError(f"no files for {dataset}/{split} below {root}")
    for path in paths:
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1024):
                yield from batch.to_pylist()
        else:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def _decoded(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _documents(pairs: list[tuple[str, list[str]]]) -> tuple[Document, ...]:
    return tuple(
        # Empty strings are retained so official sentence indices never shift.
        Document(index, clean(title), tuple(clean(x) for x in sentences))
        for index, (title, sentences) in enumerate(pairs)
    )


def _title_index(documents: tuple[Document, ...]) -> dict[str, int]:
    return {document.title.casefold(): document.doc for document in documents}


def parse_hotpot(row: dict[str, Any], split: str) -> Example:
    context = row["context"]
    documents = _documents(list(zip(context["title"], context["sentences"])))
    title_to_doc = _title_index(documents)
    facts = row.get("supporting_facts") or {"title": [], "sent_id": []}
    expected = tuple(
        dict.fromkeys(
            (clean(title).casefold(), int(sent))
            for title, sent in zip(facts.get("title", []), facts.get("sent_id", []))
        )
    )
    support = tuple(
        (title_to_doc[clean(title).casefold()], int(sent))
        for title, sent in zip(facts.get("title", []), facts.get("sent_id", []))
        if clean(title).casefold() in title_to_doc
    )
    answer = clean(row.get("answer"))
    return Example(
        dataset="hotpotqa",
        qid=str(row["id"]),
        split=split,
        question=clean(row["question"]),
        documents=documents,
        answers=(answer,) if answer else (),
        support=tuple(dict.fromkeys(support)),
        question_type=clean(row.get("type") or "unknown"),
        support_complete=len(set(support)) == len(expected),
        support_expected=len(expected),
    )


def parse_2wiki(row: dict[str, Any], split: str) -> Example:
    context = _decoded(row["context"])
    if isinstance(context, dict):
        pairs = list(zip(context["title"], context["sentences"]))
    else:
        pairs = [(item[0], item[1]) for item in context]
    documents = _documents(pairs)
    title_to_doc = _title_index(documents)
    facts = _decoded(row.get("supporting_facts", []))
    fact_pairs = list(zip(facts["title"], facts["sent_id"])) if isinstance(facts, dict) else facts
    expected = tuple(
        dict.fromkeys((clean(title).casefold(), int(sent)) for title, sent in fact_pairs)
    )
    support = tuple(
        (title_to_doc[clean(title).casefold()], int(sent))
        for title, sent in fact_pairs
        if clean(title).casefold() in title_to_doc
    )
    raw_evidences = _decoded(row.get("evidences", [])) or []
    evidences = tuple(tuple(clean(x) for x in triple[:3]) for triple in raw_evidences)
    answer = clean(row.get("answer"))
    return Example(
        dataset="2wiki",
        qid=str(row.get("_id", row.get("id"))),
        split=split,
        question=clean(row["question"]),
        documents=documents,
        answers=(answer,) if answer else (),
        support=tuple(dict.fromkeys(support)),
        question_type=clean(row.get("type") or "unknown"),
        evidences=evidences,
        support_complete=len(set(support)) == len(expected),
        support_expected=len(expected),
    )


def parse_musique(row: dict[str, Any], split: str) -> Example:
    raw_paragraphs = row["paragraphs"]
    ordered = sorted(raw_paragraphs, key=lambda item: int(item.get("idx", 0)))
    documents = tuple(
        Document(
            index,
            clean(paragraph.get("title")),
            split_sentences(clean(paragraph.get("paragraph_text"))),
        )
        for index, paragraph in enumerate(ordered)
    )
    original_to_doc = {
        int(paragraph.get("idx", index)): index for index, paragraph in enumerate(ordered)
    }
    support_docs = {
        original_to_doc[int(paragraph.get("idx", index))]
        for index, paragraph in enumerate(ordered)
        if bool(paragraph.get("is_supporting", False))
    }
    support = tuple(
        (doc, sent) for doc in sorted(support_docs) for sent in range(len(documents[doc].sentences))
    )
    decomposition: list[DecompositionStep] = []
    for index, step in enumerate(row.get("question_decomposition", []) or []):
        original = step.get("paragraph_support_idx")
        support_doc = original_to_doc.get(int(original)) if original is not None else None
        decomposition.append(
            DecompositionStep(
                step_id=index,
                question=clean(step.get("question")),
                answer=clean(step.get("answer")),
                support_doc=support_doc,
            )
        )
    answers = [clean(row.get("answer"))]
    answers.extend(clean(value) for value in row.get("answer_aliases", []) or [])
    return Example(
        dataset="musique",
        qid=str(row["id"]),
        split=split,
        question=clean(row["question"]),
        documents=documents,
        answers=tuple(dict.fromkeys(answer for answer in answers if answer)),
        support=tuple(dict.fromkeys(support)),
        question_type=f"hops={len(decomposition) or len(support_docs)}",
        decomposition=tuple(decomposition),
        support_expected=len(support),
    )


PARSERS = {"hotpotqa": parse_hotpot, "2wiki": parse_2wiki, "musique": parse_musique}


def iter_examples(
    dataset: str,
    root: Path,
    split: str,
    dataset_variant: str | None = None,
) -> Iterator[Example]:
    parser = PARSERS[dataset]
    for row in iter_raw(dataset, root, split, dataset_variant):
        yield parser(row, split)


def assert_public_record(record: dict[str, Any]) -> None:
    """Reject accidental gold leakage at the serialization boundary."""

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = key.casefold()
                if lowered in SUPERVISION_KEYS or lowered.startswith("gold"):
                    raise ValueError(f"supervision leaked into public record at {path}/{key}")
                walk(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}/{index}")

    walk(record)
