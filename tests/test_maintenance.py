from dataclasses import replace

from verijoin.compiler import compile_gold
from verijoin.lineage import bind_lineage, verify_lineage
from verijoin.maintenance import refresh_result
from verijoin.schema import AnswerExpr, Document, Example
from verijoin.vm import execute


def _example() -> Example:
    return Example(
        dataset="hotpotqa",
        qid="q1",
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


def test_refresh_reuses_replays_or_recompiles_fail_closed() -> None:
    example = _example()
    program = compile_gold(example)
    answer = execute(example, program).answer
    snapshot = bind_lineage(example, program)

    distractors = list(example.documents)
    distractors[2] = replace(distractors[2], sentences=("Changed distractor.",))
    unchanged_read_set = replace(example, documents=tuple(distractors))
    reused = refresh_result(unchanged_read_set, program, answer, snapshot)
    assert reused.action == "reuse"
    assert reused.snapshot == snapshot

    documents = list(example.documents)
    documents[1] = replace(
        documents[1],
        sentences=(documents[1].sentences[0] + " Biography updated.",),
    )
    replayable = replace(example, documents=tuple(documents))
    replayed = refresh_result(replayable, program, answer, snapshot)
    assert replayed.action == "replay"
    assert replayed.snapshot is not None
    assert verify_lineage(replayable, replayed.snapshot).current

    documents[1] = replace(
        documents[1],
        sentences=("Ada Writer was born in Counterfactual City.",),
    )
    invalid = replace(example, documents=tuple(documents))
    rejected = refresh_result(invalid, program, answer, snapshot)
    assert rejected.action == "recompile"
    assert rejected.answer == ""


def test_structure_guard_recompiles_on_sentence_insert_or_delete() -> None:
    example = _example()
    program = compile_gold(example)
    answer = execute(example, program).answer
    snapshot = bind_lineage(example, program)

    inserted_documents = list(example.documents)
    inserted_documents[2] = replace(
        inserted_documents[2],
        sentences=inserted_documents[2].sentences + ("A newly inserted fact.",),
    )
    inserted = replace(example, documents=tuple(inserted_documents))
    inserted_check = verify_lineage(inserted, snapshot)
    assert inserted_check.structure_changed
    assert refresh_result(inserted, program, answer, snapshot).action == "recompile"

    deleted_documents = list(example.documents)
    deleted_documents[2] = replace(deleted_documents[2], sentences=())
    deleted = replace(example, documents=tuple(deleted_documents))
    deleted_check = verify_lineage(deleted, snapshot)
    assert deleted_check.structure_changed
    assert refresh_result(deleted, program, answer, snapshot).action == "recompile"


def test_stale_uncertified_answer_requires_model_recompilation() -> None:
    example = _example()
    grounded = compile_gold(example)
    program = replace(grounded, answer=AnswerExpr("bool", value="yes"))
    answer = execute(example, program).answer
    snapshot = bind_lineage(example, program)
    assert not snapshot.lineage_certified

    unchanged = refresh_result(example, program, answer, snapshot)
    assert unchanged.action == "reuse"

    documents = list(example.documents)
    documents[0] = replace(documents[0], sentences=("Book X was written by Bea Writer.",))
    stale = refresh_result(replace(example, documents=tuple(documents)), program, answer, snapshot)
    assert stale.action == "recompile"
    assert stale.answer == ""
