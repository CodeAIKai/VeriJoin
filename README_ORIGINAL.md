# VeriJoin

VeriJoin is a clean-room implementation for the new VLDB direction. It does **not** implement
HopCover, summary routing, ChainClosure, or any result from the rejected/non-full experiments.

The research hypothesis is that a trained 7B model should act as a compiler rather than a free-form
reasoner. It emits a small program whose source pointers are exact quotes from displayed sentences,
document titles, or the question. The VM late-binds quotes to offsets, executes typed reducers
(`argmin`, `argmax`, `equal`, `common`), distinguishes evidence equi-joins from query-constant
joins, checks lineage connectivity, and fails closed. This makes an intermediate entity a replayable
data item instead of an unverifiable chain-of-thought string. After successful execution,
`bind_lineage` records SHA-256 versions for every question/title/sentence cell actually read;
`verify_lineage` detects a stale cached result after a relevant corpus update while ignoring changes
to unreferenced distractors.

## Evaluation contract

- Main local benchmark: every labeled official development example, not a sample.
- HotpotQA distractor validation: 7,405 questions.
- 2WikiMultiHopQA development: 12,576 questions.
- MuSiQue answerable development: 2,417 questions.
- Blind test numbers are reported only after an official leaderboard submission.
- Closed-context and open-corpus retrieval results are kept in separate tables.
- Official-style Answer EM/F1 scores the grounded answer expression independently of proof
validity, as the benchmark scorers do. Strict Answer EM/F1 additionally treats disconnected or
otherwise invalid proofs as fail-closed abstentions. Both are always reported.
- Lineage-certified coverage and conditional Answer EM/F1 are separate again: only successful
  `copy/argmin/argmax/equal/common` executions qualify. A learned `bool` can be valid and grounded
  in cited evidence, but is never mislabeled as a deterministically certified answer.
- HotpotQA and 2Wiki report sentence evidence and official-style joint metrics; MuSiQue reports
  supporting-document metrics. A fullwiki evidence score is only computed on examples whose gold
  documents occur in the supplied retrieved context, and is never mixed with distractor results.

## Quick start

```bash
cd /path/to/VeriJoin
export PYTHONPATH="$PWD/src"
python -m verijoin.cli count --raw-root /path/to/datasets --split dev
python -m verijoin.cli audit --raw-root /path/to/datasets --split dev
pytest
```

Build the deterministic train/holdout partitions:

```bash
./.venv/bin/python -m verijoin.cli build-sft \
  --raw-root /path/to/datasets \
  --output artifacts/sft/train.v3.jsonl \
  --partition train --strict
./.venv/bin/python -m verijoin.cli build-sft \
  --raw-root /path/to/datasets \
  --output artifacts/sft/holdout.v3.jsonl \
  --partition holdout --strict
./.venv/bin/python -m verijoin.cli filter-lengths \
  --model /path/to/Qwen2.5-7B-Instruct \
  --input artifacts/sft/train.v3.jsonl \
  --output artifacts/sft/train.v3.max6656.jsonl \
  --max-length 6656
```

Training is intentionally launched only after the compiler coverage audit passes. The supplied
configuration uses Qwen2.5-7B-Instruct with NF4 QLoRA, rank 64, response-only loss, BF16 compute,
operator-aware balanced sampling, and gradient checkpointing on the local 32GB RTX 5090.
`configs/pilot-qlora.json` is the short end-to-end gate; `stage1-2k-qlora.json` and
`stage2-6k-qlora.json` are learning-curve checkpoints. `stage3-12k-qlora.json` is the final staged
configuration and points to the corrected v3 files. `stage3-infer-full.json` has no example cap;
`stage3-fullwiki-full.json` is a separately labelled HotpotQA retrieval-stress run.
`stage3-guided-200.json` smoke-tests four-candidate VM-guided decoding for runtime compatibility;
its score does not determine whether the pre-registered uncapped run is reported. Candidate selection
uses only executable validity, normalized answer consensus, and model likelihood; it never reads gold
answers or supporting facts. The single-candidate greedy run is retained as the compute-matched
ablation.

The final checkpoint is selected only by the balanced 2% holdout drawn from the official training
splits. Development diagnostics are not used for checkpoint selection. The pre-registered greedy
N=1 and VM-guided N=4 settings are both evaluated on every labeled development example and both
are reported, including token and wall-clock costs; the N=4 result is not conditionally hidden when
it loses.

The final v3 length-filtered files contain 267,364 training rows and 5,437 holdout rows. A streaming
ID audit found zero train/holdout overlap and zero overlap between either file and the 22,398 labeled
development examples.

```bash
PYTHONPATH=src ./.venv/bin/python -m verijoin.cli train --config configs/pilot-qlora.json
PYTHONPATH=src ./.venv-infer/bin/python -m verijoin.cli infer \
  --config configs/pilot-infer.json --raw-root /path/to/datasets \
  --dataset hotpotqa --split dev --output artifacts/predictions/pilot-hotpotqa.jsonl
PYTHONPATH=src ./.venv/bin/python -m verijoin.cli evaluate \
  --raw-root /path/to/datasets --dataset hotpotqa --split dev \
  --predictions artifacts/predictions/pilot-hotpotqa.jsonl \
  --report artifacts/results/pilot-hotpotqa.json
```

After choosing the checkpoint from training-holdout loss and updating the two full inference configs,
`scripts/run_full_dev.sh` executes the pre-registered N=1/N=4 protocols, the zero-call likelihood
re-selection ablation, official-style metrics, strict/certified metrics, and update diagnostics for all
22,398 labeled development examples.

## Current status

The parser, typed IR, compiler, VM, versioned-lineage checker, dual-contract evaluator,
no-truncation SFT builder, staged QLoRA trainer, Transformers gate, and CUDA 13.0 vLLM batch path are
implemented. Final full-development evaluation is complete on all 22,398 examples with no skipped
or truncated inputs. The fixed execution-guided N=4 result improves Answer F1 over greedy N=1 from
75.08 to 76.24 on HotpotQA, 81.95 to 82.86 on 2Wiki, and 59.77 to 61.51 on MuSiQue. Strict
fail-closed F1 improves by 4.12, 2.38, and 6.34 points respectively. These are local development
results, not blind-test or 7B-SOTA claims.

See `docs/full_results.md` for complete metrics, run manifests, selector ablations, update costs, and
claim boundaries; `docs/literature.md` gives setting-matched external references;
`docs/method_review.md` is the acceptance audit; and `docs/correctness.md` defines the exact
certification property and its non-guarantees. Earlier 200-example reports and pre-logprob-fix N=4
diagnostics remain only as engineering history and must not be cited as final results.
