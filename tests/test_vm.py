from dataclasses import replace

import pytest

from verijoin.compiler import compile_gold
from verijoin.lineage import bind_lineage, verify_lineage, verify_result_binding
from verijoin.schema import AnswerExpr, Document, Example, JoinConstraint, Program, SpanPointer
from verijoin.vm import execute, parse_program


def bridge_example() -> Example:
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


def test_gold_program_is_span_grounded_and_executable() -> None:
    example = bridge_example()
    program = compile_gold(example)
    result = execute(example, program)
    assert result.valid, result.errors
    assert result.answer == "Delta City"
    assert "Ada Writer" in result.join_values
    assert any(join.kind == "query" and join.left.quote == "Book X" for join in program.joins)


def test_vm_rejects_hallucinated_join_anchor() -> None:
    example = bridge_example()
    bad = Program(
        version=1,
        mode="bridge",
        evidence=((0, 0), (1, 0)),
        joins=(JoinConstraint(SpanPointer(0, 0, 0, 6), 1, "sentence"),),
        answer=AnswerExpr("copy", SpanPointer(1, 0, -1, -1, "sentence", "Delta City")),
    )
    result = execute(example, bad)
    assert not result.valid
    assert result.answer == ""
    assert result.answer_valid
    assert result.candidate_answer == "Delta City"
    assert any("anchor is absent" in error for error in result.errors)


def test_citation_only_contract_skips_join_semantics_but_keeps_answer_grounding() -> None:
    example = bridge_example()
    bad = Program(
        version=2,
        mode="bridge",
        evidence=((0, 0), (1, 0)),
        joins=(JoinConstraint(SpanPointer(0, 0, 0, 6), 1, "sentence"),),
        answer=AnswerExpr("copy", SpanPointer(1, 0, -1, -1, "sentence", "Delta City")),
    )
    result = execute(
        example,
        bad,
        validate_joins=False,
        require_join_for_multidoc=False,
    )
    assert result.valid, result.errors
    assert result.answer == "Delta City"


def test_vm_rejects_uncited_answer_pointer() -> None:
    example = bridge_example()
    program = compile_gold(example)
    bad = Program(
        version=1,
        mode=program.mode,
        evidence=((0, 0),),
        joins=(),
        answer=program.answer,
    )
    result = execute(example, bad, require_join_for_multidoc=False)
    assert not result.valid
    assert "answer: pointer is not cited evidence" in result.errors


def test_round_trip_json() -> None:
    example = bridge_example()
    program = compile_gold(example)
    encoded = f"<PROGRAM>{__import__('json').dumps(program.to_dict())}</PROGRAM>"
    assert parse_program(encoded) == program


def test_parse_program_rejects_non_object_answer_fail_closed() -> None:
    with pytest.raises(TypeError, match="program.answer must be a JSON object"):
        parse_program('{"version": 2, "answer": "Delta City"}')


def test_parse_program_rejects_non_object_root_fail_closed() -> None:
    with pytest.raises(TypeError, match="program must be a JSON object"):
        parse_program('[{"answer": {"op": "literal", "value": "Delta City"}}]')


def test_comparison_answer_can_copy_a_cited_title() -> None:
    example = Example(
        dataset="2wiki",
        qid="q2",
        split="dev",
        question="Which film came first, Alpha or Beta?",
        documents=(
            Document(0, "Alpha", ("Alpha was released in 2001.",)),
            Document(1, "Beta", ("Beta was released in 1999.",)),
        ),
        answers=("Beta",),
        support=((0, 0), (1, 0)),
        question_type="comparison",
    )
    result = execute(example, compile_gold(example))
    assert result.valid, result.errors
    assert result.answer == "Beta"


def test_late_bound_quote_resolves_without_model_generated_offsets() -> None:
    example = bridge_example()
    pointer = SpanPointer(1, 0, -1, -1, "sentence", "Delta City")
    program = Program(
        version=1,
        mode="comparison",
        evidence=((1, 0),),
        joins=(),
        answer=AnswerExpr("copy", pointer),
    )
    result = execute(example, program)
    assert result.valid, result.errors
    assert result.answer == "Delta City"


def test_vm_executes_argmin_instead_of_trusting_a_selected_answer() -> None:
    example = Example(
        dataset="2wiki",
        qid="q3",
        split="dev",
        question="Which film came out first, Alpha or Beta?",
        documents=(
            Document(0, "Alpha", ("Alpha was released in 2001.",)),
            Document(1, "Beta", ("Beta was released in 1999.",)),
        ),
        support=((0, 0), (1, 0)),
    )
    answer = AnswerExpr(
        "argmin",
        operands=(
            SpanPointer(0, 0, -1, -1, "sentence", "2001"),
            SpanPointer(1, 0, -1, -1, "sentence", "1999"),
        ),
        labels=(
            SpanPointer(0, -1, -1, -1, "title", "Alpha"),
            SpanPointer(1, -1, -1, -1, "title", "Beta"),
        ),
        value_type="date",
    )
    result = execute(example, Program(2, "comparison", example.support, (), answer))
    assert result.valid, result.errors
    assert result.answer == "Beta"


def test_vm_executes_grounded_equality() -> None:
    example = Example(
        dataset="2wiki",
        qid="q4",
        split="dev",
        question="Are both people from the same country?",
        documents=(
            Document(0, "A", ("A is American.",)),
            Document(1, "B", ("B is from the United States.",)),
        ),
        support=((0, 0), (1, 0)),
    )
    answer = AnswerExpr(
        "equal",
        operands=(
            SpanPointer(0, 0, -1, -1, "sentence", "American"),
            SpanPointer(1, 0, -1, -1, "sentence", "United States"),
        ),
    )
    result = execute(example, Program(2, "comparison", example.support, (), answer))
    assert result.valid, result.errors
    assert result.lineage_certified
    assert result.answer == "yes"


def test_learned_boolean_is_valid_but_not_lineage_certified() -> None:
    example = bridge_example()
    program = Program(
        2,
        "comparison",
        ((0, 0),),
        (),
        AnswerExpr("bool", value="yes"),
    )
    result = execute(example, program)
    assert result.valid, result.errors
    assert not result.lineage_certified
    assert result.answer == "yes"


def test_vm_rejects_disconnected_composition_join_graph() -> None:
    example = Example(
        dataset="musique",
        qid="q5",
        split="dev",
        question="Where did the author work?",
        documents=(
            Document(0, "Book", ("Book was written by Ada.",)),
            Document(1, "Ada", ("Ada worked for Acme.",)),
            Document(2, "Acme", ("Acme is in Paris.",)),
        ),
        support=((0, 0), (1, 0), (2, 0)),
    )
    program = Program(
        2,
        "composition",
        example.support,
        (JoinConstraint(SpanPointer(0, 0, -1, -1, "sentence", "Ada"), 1, "sentence"),),
        AnswerExpr("copy", SpanPointer(2, 0, -1, -1, "sentence", "Paris")),
    )
    result = execute(example, program)
    assert not result.valid
    assert "multi-document proof join graph is disconnected" in result.errors


def test_vm_rejects_out_of_range_join_document_without_raising() -> None:
    example = bridge_example()
    program = Program(
        2,
        "bridge",
        ((0, 0), (99, 0)),
        (JoinConstraint(SpanPointer(0, 0, -1, -1, "sentence", "Ada Writer"), 99),),
        AnswerExpr("copy", SpanPointer(0, 0, -1, -1, "sentence", "Book X")),
    )
    result = execute(example, program)
    assert not result.valid
    assert "join[0]: right document out of range" in result.errors


def test_bound_lineage_detects_relevant_but_not_irrelevant_updates() -> None:
    original = bridge_example()
    documents = list(original.documents)
    documents[1] = replace(
        documents[1],
        sentences=documents[1].sentences + ("An unrelated sentence in a joined document.",),
    )
    example = replace(original, documents=tuple(documents))
    snapshot = bind_lineage(example, compile_gold(example))
    assert len(snapshot.program_sha256) == 64
    assert len(snapshot.answer_sha256) == 64
    assert snapshot.vm_semantics_version == "verijoin-vm-2"
    assert snapshot.lineage_certified
    assert verify_result_binding(snapshot, compile_gold(example), "Delta City")
    assert not verify_result_binding(snapshot, compile_gold(example), "New City")

    unrelated_documents = list(example.documents)
    unrelated_documents[2] = replace(
        unrelated_documents[2], sentences=("A changed distractor.",)
    )
    assert verify_lineage(replace(example, documents=tuple(unrelated_documents)), snapshot).current

    joined_documents = list(example.documents)
    joined_documents[1] = replace(
        joined_documents[1],
        sentences=(joined_documents[1].sentences[0], "A changed but unread joined sentence."),
    )
    assert verify_lineage(replace(example, documents=tuple(joined_documents)), snapshot).current

    changed_documents = list(example.documents)
    changed_documents[1] = replace(
        changed_documents[1],
        sentences=("Ada Writer was born in New City.", changed_documents[1].sentences[1]),
    )
    check = verify_lineage(replace(example, documents=tuple(changed_documents)), snapshot)
    assert not check.current
    assert ("sentence", 1, 0) in check.changed


def test_lineage_refuses_to_bind_an_invalid_program() -> None:
    example = bridge_example()
    program = Program(
        2,
        "comparison",
        (),
        (),
        AnswerExpr("copy", SpanPointer(1, 0, -1, -1, "sentence", "Delta City")),
    )
    import pytest

    with pytest.raises(ValueError, match="cannot bind lineage"):
        bind_lineage(example, program)
