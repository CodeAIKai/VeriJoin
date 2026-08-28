import numpy as np

from verijoin.schema import Document, Example
from verijoin.significance import _paired_bootstrap, _task_score


def test_paired_bootstrap_reports_candidate_minus_baseline_direction() -> None:
    report = _paired_bootstrap(
        np.asarray((1.0, 1.0, 1.0)),
        samples=100,
        seed=7,
    )

    assert report["mean_difference"] == 100.0
    assert report["ci95_lower"] == 100.0
    assert report["ci95_upper"] == 100.0
    assert report["candidate_wins"] == 3
    assert report["candidate_losses"] == 0
    assert report["two_sided_p"] == 2 / 101


def test_task_score_parses_answer_and_citation_protocols() -> None:
    example = Example(
        dataset="hotpotqa",
        qid="q1",
        split="dev",
        question="Where?",
        documents=(Document(0, "Place", ("Delta City is here.",)),),
        answers=("Delta City",),
        support=((0, 0),),
        question_type="bridge",
    )

    answer, eligible = _task_score(
        example, {"output": "<ANSWER>Delta City</ANSWER>"}, "answer"
    )
    citation, citation_eligible = _task_score(
        example,
        {"output": '<CITATION>{"answer":"Delta City","evidence":[[0,0]]}</CITATION>'},
        "citation",
    )

    assert eligible and citation_eligible
    assert answer == 1.0
    assert citation == 1.0
