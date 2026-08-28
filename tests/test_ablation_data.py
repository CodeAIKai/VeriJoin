from verijoin.ablation_data import answer_from_program
from verijoin.schema import AnswerExpr, Program, SpanPointer


def _pointer(value: str) -> SpanPointer:
    return SpanPointer(0, 0, -1, -1, "sentence", value)


def test_answer_label_executes_ordered_program_instead_of_copying_operand() -> None:
    program = Program(
        2,
        "comparison",
        ((0, 0), (1, 0)),
        (),
        AnswerExpr(
            "argmin",
            operands=(_pointer("2001"), _pointer("1999")),
            labels=(_pointer("Alpha"), _pointer("Beta")),
            value_type="date",
        ),
    )
    assert answer_from_program(program) == "Beta"


def test_answer_label_uses_same_canonical_equality_as_vm() -> None:
    program = Program(
        2,
        "comparison",
        ((0, 0), (1, 0)),
        (),
        AnswerExpr("equal", operands=(_pointer("American"), _pointer("United States"))),
    )
    assert answer_from_program(program) == "yes"
