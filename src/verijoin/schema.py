from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DatasetName = Literal["hotpotqa", "2wiki", "musique"]


@dataclass(frozen=True, slots=True)
class Sentence:
    doc: int
    sent: int
    text: str

    @property
    def ref(self) -> tuple[int, int]:
        return self.doc, self.sent


@dataclass(frozen=True, slots=True)
class Document:
    doc: int
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass(frozen=True, slots=True)
class DecompositionStep:
    step_id: int
    question: str
    answer: str
    support_doc: int | None = None


@dataclass(frozen=True, slots=True)
class Example:
    dataset: DatasetName
    qid: str
    split: str
    question: str
    documents: tuple[Document, ...]
    answers: tuple[str, ...] = ()
    support: tuple[tuple[int, int], ...] = ()
    question_type: str = "unknown"
    decomposition: tuple[DecompositionStep, ...] = ()
    evidences: tuple[tuple[str, str, str], ...] = ()
    support_complete: bool = True
    support_expected: int = 0

    def sentence(self, ref: tuple[int, int]) -> Sentence:
        doc_id, sent_id = ref
        document = self.documents[doc_id]
        return Sentence(doc_id, sent_id, document.sentences[sent_id])

    def public_dict(self) -> dict[str, Any]:
        """Return inference input with every supervision field deliberately omitted."""
        return {
            "dataset": self.dataset,
            "qid": self.qid,
            "split": self.split,
            "question": self.question,
            "documents": [
                {"doc": doc.doc, "title": doc.title, "sentences": list(doc.sentences)}
                for doc in self.documents
            ],
        }


@dataclass(frozen=True, slots=True)
class SpanPointer:
    doc: int
    sent: int
    start: int
    end: int
    field: Literal["sentence", "title", "question"] = "sentence"
    quote: str = ""

    def ref(self) -> tuple[int, int]:
        return self.doc, self.sent


@dataclass(frozen=True, slots=True)
class JoinConstraint:
    left: SpanPointer
    right_doc: int
    right_field: Literal["title", "sentence"] = "title"
    kind: Literal["equi", "query"] = "equi"


@dataclass(frozen=True, slots=True)
class AnswerExpr:
    op: Literal["copy", "bool", "literal", "argmin", "argmax", "equal", "common"]
    pointer: SpanPointer | None = None
    value: str = ""
    operands: tuple[SpanPointer, ...] = ()
    labels: tuple[SpanPointer, ...] = ()
    value_type: Literal["text", "number", "date"] = "text"


@dataclass(frozen=True, slots=True)
class Program:
    version: int
    mode: Literal["bridge", "comparison", "composition", "unknown"]
    evidence: tuple[tuple[int, int], ...]
    joins: tuple[JoinConstraint, ...]
    answer: AnswerExpr
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = [list(ref) for ref in self.evidence]
        return result


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    lineage_certified: bool
    answer: str
    candidate_answer: str
    answer_valid: bool
    errors: tuple[str, ...]
    cited_text: tuple[str, ...]
    join_values: tuple[str, ...]
