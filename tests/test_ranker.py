import json
from dataclasses import replace

from verijoin.meta_ranker import FEATURE_NAMES, _checkpoint_score, answer_groups
from verijoin.ranker import RankCandidate, _select_ranked, candidate_records
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


def _record(index: int, answer: str, tier: int) -> RankCandidate:
    return RankCandidate(index, "raw", answer, tier, "passage", -0.1)


def test_ranked_selection_picks_best_answer_then_valid_program() -> None:
    records = [_record(0, "Alpha", 2), _record(1, "Beta", 1), _record(2, "Beta", 2)]
    assert _select_ranked(records, [0.0, 2.0, 1.0]) == 2


def test_candidate_rendering_is_independent_of_gold_labels() -> None:
    example = Example(
        dataset="hotpotqa",
        qid="rank",
        split="dev",
        question="Which token?",
        documents=(Document(0, "Tokens", ("Alpha and Beta are tokens.",)),),
        answers=("Beta",),
        support=((0, 0),),
    )
    row = {
        "candidates": [
            {"output": _copy("Beta"), "average_logprob": -0.2},
            {"output": _copy("Alpha"), "average_logprob": -0.3},
        ]
    }
    unlabeled = replace(example, answers=(), support=())
    assert candidate_records(example, row) == candidate_records(unlabeled, row)


def test_answer_group_features_are_independent_of_gold_labels() -> None:
    example = Example(
        dataset="hotpotqa",
        qid="groups",
        split="dev",
        question="Which token?",
        documents=(Document(0, "Tokens", ("Alpha and Beta are tokens.",)),),
        answers=("Beta",),
        support=((0, 0),),
    )
    row = {
        "candidates": [
            {
                "output": _copy("Beta"),
                "average_logprob": -0.2,
                "ranker_score": 1.0,
                "generated_tokens": 10,
            },
            {
                "output": _copy("Alpha"),
                "average_logprob": -0.3,
                "ranker_score": 0.0,
                "generated_tokens": 11,
            },
        ]
    }
    unlabeled = replace(example, answers=(), support=())
    assert answer_groups(example, row) == answer_groups(unlabeled, row)


def test_manual_meta_checkpoint_scores_features() -> None:
    size = len(FEATURE_NAMES)
    checkpoint = {
        "means": [0.0] * size,
        "scales": [1.0] * size,
        "weight1": [[1.0] + [0.0] * (size - 1)],
        "bias1": [0.0],
        "weight2": [1.0],
        "bias2": 0.0,
    }
    low = (0.25,) + (0.0,) * (size - 1)
    high = (0.75,) + (0.0,) * (size - 1)
    assert _checkpoint_score(high, checkpoint) > _checkpoint_score(low, checkpoint)
