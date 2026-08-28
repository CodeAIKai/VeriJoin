import json

from verijoin.candidate_eval import analyze_candidate_row
from verijoin.schema import AnswerExpr, Document, Example, Program, SpanPointer


def _copy(quote: str, evidence: tuple[tuple[int, int], ...] = ((0, 0),)) -> str:
    program = Program(
        2,
        "bridge",
        evidence,
        (),
        AnswerExpr("copy", SpanPointer(0, 0, -1, -1, "sentence", quote)),
    )
    return json.dumps(program.to_dict())


def test_candidate_analysis_separates_oracle_from_consensus() -> None:
    example = Example(
        dataset="hotpotqa",
        qid="oracle",
        split="dev",
        question="Which token?",
        documents=(Document(0, "Tokens", ("Alpha and Beta are tokens.",)),),
        answers=("Beta",),
        support=((0, 0),),
    )
    row = {
        "output": _copy("Alpha"),
        "candidates": [
            {"output": _copy("Alpha")},
            {"output": _copy("Alpha")},
            {"output": _copy("Beta")},
        ],
    }
    analysis = analyze_candidate_row(example, row)
    assert analysis.selected_f1 == 0.0
    assert analysis.oracle_em == 1.0
    assert analysis.oracle_f1 == 1.0
    assert analysis.global_consensus_f1 == 0.0
