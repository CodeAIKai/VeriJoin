from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from collections.abc import Iterable
from html import unescape

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def clean(text: object) -> str:
    return " ".join(unescape(str(text or "")).split())


def split_sentences(text: str) -> tuple[str, ...]:
    value = clean(text)
    if not value:
        return ()
    pieces = tuple(piece.strip() for piece in _SENTENCE_BOUNDARY.split(value) if piece.strip())
    return pieces or (value,)


def normalize_answer(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).casefold()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = _ARTICLES.sub(" ", value)
    return " ".join(value.split())


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(answer) for answer in answers))


def token_prf(prediction: str, answers: Iterable[str]) -> tuple[float, float, float]:
    """Return precision/recall/F1 for the best matching answer alias."""
    pred_tokens = normalize_answer(prediction).split()
    best = (0.0, 0.0, 0.0)
    for answer in answers:
        gold_tokens = normalize_answer(answer).split()
        if (
            pred_tokens != gold_tokens
            and ({"yes", "no"} & ({" ".join(pred_tokens), " ".join(gold_tokens)}))
        ):
            candidate = (0.0, 0.0, 0.0)
        elif not pred_tokens or not gold_tokens:
            score = float(pred_tokens == gold_tokens)
            candidate = (score, score, score)
        else:
            overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
            if overlap == 0:
                candidate = (0.0, 0.0, 0.0)
            else:
                precision = overlap / len(pred_tokens)
                recall = overlap / len(gold_tokens)
                score = 2 * precision * recall / (precision + recall)
                candidate = (precision, recall, score)
        if candidate[2] > best[2]:
            best = candidate
    return best


def token_f1(prediction: str, answers: Iterable[str]) -> float:
    return token_prf(prediction, answers)[2]


def find_span(text: str, candidates: Iterable[str]) -> tuple[int, int, str] | None:
    """Find a literal source span, preferring the longest answer alias."""
    folded = text.casefold()
    for candidate in sorted(set(candidates), key=len, reverse=True):
        value = clean(candidate)
        if not value:
            continue
        start = folded.find(value.casefold())
        if start >= 0:
            return start, start + len(value), text[start : start + len(value)]
    return None


def normalized_contains(haystack: str, needle: str) -> bool:
    target = normalize_answer(needle)
    return bool(target and target in normalize_answer(haystack))
