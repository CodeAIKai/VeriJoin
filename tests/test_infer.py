import json
from dataclasses import replace
from pathlib import Path

import pytest

from verijoin import infer as infer_module
from verijoin.infer import (
    InferenceConfig,
    InferenceSummary,
    _average_logprob,
    _select_candidate,
    run_inference,
)
from verijoin.prompt import inference_messages
from verijoin.schema import AnswerExpr, Document, Example, Program, SpanPointer


def _example() -> Example:
    return Example(
        dataset="hotpotqa",
        qid="decode",
        split="dev",
        question="Which token?",
        documents=(Document(0, "Tokens", ("Alpha and Beta are tokens.",)),),
        support=((0, 0),),
    )


def _copy(quote: str, evidence: tuple[tuple[int, int], ...] = ((0, 0),)) -> str:
    program = Program(
        2,
        "comparison",
        evidence,
        (),
        AnswerExpr("copy", SpanPointer(0, 0, -1, -1, "sentence", quote)),
    )
    return json.dumps(program.to_dict())


def test_execution_guided_selection_prefers_strictly_valid_candidate() -> None:
    selected, metadata = _select_candidate(
        _example(),
        [(_copy("Alpha", evidence=((0, 0), (2, 0))), -0.01), (_copy("Beta"), -0.2)],
        execution_guided=True,
    )
    assert selected == 1
    assert metadata["strict_valid_candidates"] == 1
    assert metadata["answer_valid_candidates"] == 2


def test_execution_guided_selection_uses_answer_consensus_before_likelihood() -> None:
    selected, metadata = _select_candidate(
        _example(),
        [(_copy("Alpha"), -0.2), (_copy("Alpha"), -0.3), (_copy("Beta"), -0.01)],
        execution_guided=True,
    )
    assert selected == 0
    assert metadata["selected_answer_consensus"] == 2


def test_likelihood_selection_can_be_used_as_an_ablation() -> None:
    selected, metadata = _select_candidate(
        _example(),
        [(_copy("Alpha"), -0.2), (_copy("Beta"), -0.01)],
        execution_guided=False,
    )
    assert selected == 1
    assert metadata["selected_tier"] == 0


def test_answer_protocol_uses_parse_and_consensus_without_labels() -> None:
    selected, metadata = _select_candidate(
        _example(),
        [
            ("<ANSWER>Alpha</ANSWER>", -0.2),
            ("<ANSWER>Alpha</ANSWER>", -0.3),
            ("<ANSWER>Beta</ANSWER>", -0.01),
            ("unparseable", -0.001),
        ],
        execution_guided=True,
        task="answer",
    )
    assert selected == 0
    assert metadata["selected_answer_consensus"] == 2
    assert metadata["answer_valid_candidates"] == 3


def test_citation_protocol_prefers_in_range_evidence() -> None:
    selected, metadata = _select_candidate(
        _example(),
        [
            ('<CITATION>{"answer":"Alpha","evidence":[[4,0]]}</CITATION>', -0.01),
            ('<CITATION>{"answer":"Beta","evidence":[[0,0]]}</CITATION>', -0.2),
        ],
        execution_guided=True,
        task="citation",
    )
    assert selected == 1
    assert metadata["strict_valid_candidates"] == 1
    assert metadata["answer_valid_candidates"] == 2



def test_citation_protocol_requires_nonempty_evidence_for_top_tier() -> None:
    empty = '<CITATION>{"answer":"Alpha","evidence":[]}</CITATION>'
    grounded = '<CITATION>{"answer":"Alpha","evidence":[[0,0]]}</CITATION>'
    selected, metadata = _select_candidate(
        _example(),
        [(empty, -0.01), (grounded, -0.2)],
        execution_guided=True,
        task="citation",
    )
    assert selected == 1
    assert metadata["selected_tier"] == 2

def test_citation_protocol_fails_closed_on_malformed_reference() -> None:
    malformed = '<CITATION>{"answer":"Alpha","evidence":[[0]]}</CITATION>'
    grounded = '<CITATION>{"answer":"Alpha","evidence":[[0,0]]}</CITATION>'
    selected, metadata = _select_candidate(
        _example(),
        [(malformed, -0.01), (grounded, -0.2)],
        execution_guided=True,
        task="citation",
    )
    assert selected == 1
    assert metadata["strict_valid_candidates"] == 1


def test_free_literal_protocol_executes_with_literal_enabled() -> None:
    literal = Program(
        2,
        "comparison",
        ((0, 0),),
        (),
        AnswerExpr("literal", value="Alpha"),
    )
    selected, metadata = _select_candidate(
        _example(),
        [(json.dumps(literal.to_dict()), -0.2), ("broken", -0.01)],
        execution_guided=True,
        task="free_literal",
    )
    assert selected == 0
    assert metadata["strict_valid_candidates"] == 1


def test_average_logprob_rejects_missing_vllm_scores() -> None:
    assert _average_logprob(-4.0, 2) == -2.0
    with pytest.raises(RuntimeError, match="did not return cumulative"):
        _average_logprob(None, 2)


def test_execution_guided_selection_is_independent_of_gold_labels() -> None:
    example = _example()
    candidates = [(_copy("Alpha"), -0.2), (_copy("Beta"), -0.1)]
    labeled = replace(example, answers=("Alpha",), support=((0, 0),))
    labels_removed = replace(example, answers=(), support=())
    assert _select_candidate(labeled, candidates, execution_guided=True) == _select_candidate(
        labels_removed, candidates, execution_guided=True
    )


def test_multi_candidate_config_rejects_greedy_temperature(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model": "m", "adapter": None, "num_candidates": 4}))
    with pytest.raises(ValueError, match="temperature"):
        InferenceConfig.from_json(path)


def test_free_literal_inference_uses_training_contract_prompt() -> None:
    messages = inference_messages(_example(), "free_literal")
    assert "answer must use op=literal" in messages[0]["content"]
    assert "op=literal answer" in messages[1]["content"]


def test_inference_refuses_unmanifested_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "partial.jsonl"
    output.write_text('{"id":"partial"}\n')
    config = InferenceConfig(model="unused", adapter=None, backend="transformers")
    with pytest.raises(ValueError, match="without a provenance manifest"):
        run_inference(config, "hotpotqa", tmp_path, "dev", output)


def test_noop_resume_preserves_completed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "complete.jsonl"
    output.write_text('{"id":"done"}\n')
    config = InferenceConfig(model="unused", adapter=None, backend="transformers")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    original = {
        "config": config.__dict__ if hasattr(config, "__dict__") else {
            field: getattr(config, field) for field in config.__dataclass_fields__
        },
        "dataset": "hotpotqa",
        "split": "dev",
        "status": "complete",
        "summary": {"generation_seconds": 123.0, "new_output_tokens": 456},
    }
    manifest_path.write_text(json.dumps(original))

    def fake_run(*args: object, **kwargs: object) -> InferenceSummary:
        return InferenceSummary(
            dataset="hotpotqa",
            dataset_variant="distractor",
            split="dev",
            output=str(output),
            total_examples=1,
            previously_complete=1,
            newly_written=0,
            new_input_tokens=0,
            new_output_tokens=0,
            render_seconds=0.0,
            engine_init_seconds=0.0,
            generation_seconds=0.0,
            examples_per_second=0.0,
            output_tokens_per_second=0.0,
            skipped_overlength=0,
            skipped_overlength_ids=(),
        )

    monkeypatch.setattr(infer_module, "run_transformers", fake_run)
    run_inference(config, "hotpotqa", tmp_path, "dev", output)
    assert json.loads(manifest_path.read_text()) == original
