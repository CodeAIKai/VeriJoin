from __future__ import annotations

import json

from .schema import Example, Program

SYSTEM_PROMPT = """You are the compiler for VeriJoin, a text query virtual machine.
Return exactly one <PROGRAM> JSON object. You may cite only displayed [D#:S#] sentences.
Every source pointer must copy an exact quote from its cited sentence or document title; the VM
binds that quote to character offsets. Use argmin/argmax/equal/common when the answer can be
computed from grounded operands; their result is executed by the VM, not written as a literal. A bridge or
composition over multiple documents must include a JOIN whose left span occurs in the right
document. An equi JOIN reads a cited sentence; a query JOIN reads an exact phrase from the
Question using field=question and doc=sent=-1. Never invent an intermediate value or an answer."""

FREE_LITERAL_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Never invent an intermediate value or an answer.",
    "The answer must use op=literal with the supervised answer value.",
)

ANSWER_SYSTEM_PROMPT = """Answer the multi-hop question from the displayed documents.
Return exactly one <ANSWER>answer text</ANSWER> and nothing else."""

CITATION_SYSTEM_PROMPT = """Answer the multi-hop question and cite the displayed evidence.
Return exactly one <CITATION> JSON object with keys answer and evidence. Evidence must be a list
of [document, sentence] integer pairs referring only to displayed [D#:S#] sentences."""


def render_context(example: Example, instruction: str = "Compile an executable evidence program.") -> str:
    lines = [f"Question: {example.question}", "Documents:"]
    for document in example.documents:
        lines.append(f"[D{document.doc}:TITLE] {document.title}")
        for sent, sentence in enumerate(document.sentences):
            lines.append(f"[D{document.doc}:S{sent}] {sentence}")
    lines.append(instruction)
    return "\n".join(lines)


def render_target(program: Program, *, include_metadata: bool = False) -> str:
    payload = program.to_dict()
    if not include_metadata:
        payload.pop("metadata", None)

    def late_bind(value: object) -> None:
        if isinstance(value, dict):
            if value.get("quote"):
                value.pop("start", None)
                value.pop("end", None)
            for child in value.values():
                late_bind(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                late_bind(child)

    late_bind(payload)
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"<PROGRAM>{body}</PROGRAM>"


def messages(example: Example, target: Program | None = None) -> list[dict[str, str]]:
    result = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_context(example)},
    ]
    if target is not None:
        result.append({"role": "assistant", "content": render_target(target)})
    return result


def inference_messages(example: Example, task: str = "program") -> list[dict[str, str]]:
    if task == "answer":
        system = ANSWER_SYSTEM_PROMPT
        instruction = "Return the answer."
    elif task == "citation":
        system = CITATION_SYSTEM_PROMPT
        instruction = "Return the answer and cited evidence."
    elif task == "free_literal":
        system = FREE_LITERAL_SYSTEM_PROMPT
        instruction = "Compile an executable evidence program with an op=literal answer."
    else:
        system = SYSTEM_PROMPT
        instruction = "Compile an executable evidence program."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render_context(example, instruction)},
    ]
