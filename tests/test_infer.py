import json
from dataclasses import replace
from pathlib import Path

import pytest

from verijoin.infer import InferenceConfig, _average_logprob, _select_candidate, run_inference
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
