from dataclasses import replace

from verijoin.compiler import compile_gold
from verijoin.schema import Document, Example
from verijoin.update_stress import evaluate_example_updates


def _example() -> Example:
    return Example(
        dataset="hotpotqa",
        qid="update-stress",
        split="dev",
        question="Where was the author of Book X born?",
        documents=(
            Document(0, "Book X", ("Book X was written by Ada Writer.",)),
            Document(1, "Ada Writer", ("Ada Writer was born in Delta City.",)),
            Document(2, "Distractor", ("Nothing useful is here.",)),
        ),
        answers=("Delta City",),
        support=((0, 0), (1, 0)),
        question_type="bridge",
    )


def test_update_matrix_routes_all_supported_classes_fail_closed() -> None:
    example = _example()
    program = compile_gold(example)
    result = evaluate_example_updates(example, program)

    assert result["unread_cell_update"]["reuse"] == 1
    assert result["value_preserving_rewrite"]["replay"] == 1
    assert result["answer_value_replacement"]["recompile"] == 1
    assert result["answer_value_replacement"]["oracle_repair_valid"] == 1
    assert result["answer_value_replacement"]["oracle_answer_changed"] == 1
    assert result["sentence_insertion"]["recompile"] == 1
    assert result["sentence_deletion"]["recompile"] == 1


def test_update_matrix_requires_certified_program() -> None:
    example = replace(_example(), answers=("yes",))
    program = replace(compile_gold(example), answer=replace(compile_gold(example).answer, op="bool"))
    assert evaluate_example_updates(example, program) == {}
