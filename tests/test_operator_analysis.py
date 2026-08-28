import json

from verijoin import operator_analysis
from verijoin.compiler import compile_gold
from verijoin.operator_analysis import analyze_operator_coverage
from verijoin.prompt import render_target
from verijoin.schema import Document, Example


def test_operator_analysis_groups_typed_answer_and_join_family(tmp_path, monkeypatch) -> None:
    example = Example(
        dataset="hotpotqa",
        qid="q1",
        split="dev",
        question="Where was the author of Book X born?",
        documents=(
            Document(0, "Book X", ("Book X was written by Ada Writer.",)),
            Document(1, "Ada Writer", ("Ada Writer was born in Delta City.",)),
        ),
        answers=("Delta City",),
        support=((0, 0), (1, 0)),
        question_type="bridge",
    )
    prediction = tmp_path / "predictions.jsonl"
    prediction.write_text(
        json.dumps({"id": "q1", "output": render_target(compile_gold(example))}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        operator_analysis,
        "iter_examples",
        lambda *_args, **_kwargs: iter((example,)),
    )

    report = analyze_operator_coverage("hotpotqa", tmp_path, "dev", prediction)

    assert report["parsed"] == 1
    assert report["by_answer_operator"]["copy"]["count"] == 1
    assert report["by_join_family"]["both"]["count"] == 1
    assert report["certified_rate"] == 100.0
