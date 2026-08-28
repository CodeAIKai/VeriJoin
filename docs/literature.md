# Literature and comparison contract (checked 2026-08-25)

## Numbers that answer “what is SOTA?”

There is no single comparable SOTA across these three datasets. The public numbers mix blind test
and development splits, closed distractor contexts and open-corpus retrieval, encoder readers and
7B generators, and different evidence metrics. They must remain in separate tables.

| Dataset | Strong published/official number | Setting | How it may be used |
|---|---:|---|---|
| HotpotQA | Beam Retrieval answer F1 85.04 | official blind distractor test; supervised DeBERTa reader | absolute reference only |
| 2WikiMultiHopQA | Beam Retrieval answer F1 90.87 | official blind test | absolute reference only |
| MuSiQue-Ans | Beam Retrieval answer F1 69.2 | official test leaderboard result; supervised DeBERTa reader | absolute reference only |

Primary sources: the [HotpotQA official page](https://hotpotqa.github.io/),
[2Wiki official repository](https://github.com/Alab-NII/2wikimultihop),
[Beam Retrieval (NAACL 2024)](https://aclanthology.org/2024.naacl-long.96/). The same paper reports
all three blind-test values in one consistent supervised-reader setup. Later development-set or
different-protocol values, such as those in [PEI](https://aclanthology.org/2024.lrec-main.1154/),
belong in a separate table rather than silently replacing leaderboard numbers.

## Same-scale neighbors

| Work | Backbone/training | HotpotQA | 2Wiki | MuSiQue | Metric/protocol warning |
|---|---|---:|---:|---:|---|
| Interact-RAG, ICLR 2026 | Qwen2.5-7B, SFT+RL, 12K | 61.6 | 71.0 | 39.5 | open-corpus interactive retrieval, not our closed full context |
| CRAFT, archived arXiv v1 best trace | Qwen2.5-7B, SFT+GRPO, 20K; external 30B judge | 80.10 | 84.18 | 63.06 | mean of ten 1K-example runs, not complete dev; version-pinned historical reference |
| PyRAG-RL | Qwen2.5 7B-scale agents, GRPO, 87,925; 8 A100-80GB | 40.5 EM | 49.4 EM | 20.7 EM | open-corpus E5 retrieval; complete eval splits; EM, not F1 |

Sources: [Interact-RAG (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b61f288da3c106f65d57b0d45b470b6b-Abstract-Conference.html)
and the version-pinned [CRAFT arXiv v1 PDF](https://arxiv.org/pdf/2602.01348v1). The archived CRAFT
table orders columns as MuSiQue, HotpotQA, and 2Wiki; values above are reordered by dataset. Its
setup samples 20K training entries, uses SFT plus GRPO with a Qwen3-30B-A3B judge, and reports ten
random 1K-example evaluation runs. The [current CRAFT record](https://arxiv.org/abs/2602.01348) was
substantially revised in July 2026 and is now titled “Does Faithfulness-Guided Alignment Hurt
Accuracy?”; therefore the archived values are a historical, version-pinned target, not a claim about
the newest manuscript or an authoritative 7B leaderboard. Unlike settings must not be placed in one
ranked column.

[PyRAG](https://arxiv.org/abs/2605.12975) is a particularly close May-2026 preprint. It executes
generated Python over neural `retrieve()` and `answer()` tools, adds compiler-error repair and
adaptive retrieval, and reports complete 7,405/12,576/2,417 evaluation splits. Its Appendix E uses
87,925 training examples and 8 A100-80GB GPUs. Its open-corpus EM numbers belong in a separate
setting, but its existence means executable programs, compiler feedback, and self-repair are not
available as standalone novelty claims for VeriJoin.

## Why the selected idea is narrower than existing systems

- [AOP (CIDR 2025)](https://www.vldb.org/cidrdb/papers/2025/p32-wang.pdf) and CAESURA already cover
  declarative planning and optimization for LLM-powered analytics. A generic operator DAG is not new.
- [LOTUS (PVLDB)](https://www.vldb.org/pvldb/vol18/p4171-patel.pdf) provides semantic operators and
  accuracy/cost optimization. VeriJoin instead asks whether generated intermediate values can be
  prohibited unless they possess replayable source lineage.
- [DocETL (PVLDB)](https://www.vldb.org/pvldb/vol18/p3035-shankar.pdf) optimizes LLM document
  pipelines; it is not an answer-lineage VM.
- [TRAQ (NAACL 2024)](https://aclanthology.org/2024.naacl-long.210/) covers conformal correctness
  guarantees for RAG. End-to-end calibration alone is therefore not the contribution here.
- [FLARE (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1193/) already plans with logic
  programs, but asks the LLM itself to simulate their execution. VeriJoin must demonstrate that an
  external typed VM, exact source binding, and fail-closed semantics change measurable behavior.
- CRAFT already covers structured faithful traces and judge rewards. VeriJoin must win on executable
  replay, deterministic reductions, fail-closed behavior, counterfactual consistency, and systems
  cost—not on the claim that JSON traces are more faithful.
- [GroundedCache (2026 preprint)](https://arxiv.org/abs/2605.27494) already combines source versions,
  evidence overlap, and answer-support gates for safe answer-cache reuse. VeriJoin's source snapshot
  is supporting machinery for executable query results, not a standalone novelty claim.
- [BLIP (VLDB 2026)](https://yiminglin18.com/publication/blip/) already gives bolt-on verifiable
  provenance by finding a small input subset that reproduces a black-box LLM result. VeriJoin cannot
  claim provenance alone. Its narrower hypothesis is that deterministic typed results can carry
  byte-exact value lineage at execution time, without repeated black-box calls; learned `bool`
  results are explicitly outside that certified subset.

## Declared local evaluation

- Qwen2.5-7B-Instruct, local NF4 QLoRA; no commercial API.
- One common compiler and VM across all datasets; no dataset-specific inference branch.
- Every labeled development example: HotpotQA 7,405, 2Wiki 12,576, MuSiQue-answerable 2,417;
  total 22,398.
- Complete provided context with no truncation (input cap 6,144; generated program cap 384;
  model cap 6,656).
- Official-style Answer EM/F1 scores an executable grounded answer expression independently from
  its proof validity, matching benchmark scorers. Malformed, answer-invalid, literal, and missing
  outputs score as wrong. A separate strict Answer EM/F1 treats every disconnected or otherwise
  invalid proof as fail-closed abstention.
- HotpotQA and 2Wiki additionally report sentence evidence EM/P/R/F1 and official-style joint
  EM/F1; MuSiQue reports supporting-document EM/P/R/F1.
- These are full development results, not official blind-test SOTA. A hidden-test claim requires an
  official submission.

## Non-final learning-curve checkpoint

The 2k QLoRA checkpoint is a diagnostic only: 200 deterministic BF16 generations from the start of
each development split, not a random sample or full result.

| Dataset | Valid % | Answer EM/F1 | Strict EM/F1 | Evidence F1 | Joint F1 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 72.5 | 54.0 / 68.09 | 43.5 / 53.95 | 82.57 sentence | 59.70 |
| 2WikiMultiHopQA | 93.5 | 66.5 / 75.51 | 66.0 / 74.48 | 93.07 sentence | 73.11 |
| MuSiQue-Ans | 62.5 | 49.0 / 61.26 | 38.0 / 46.96 | 83.67 document | n/a |

The conditional answer F1 among valid programs is 74.41, 79.66, and 75.13 respectively. Thus the
dominant early-checkpoint error is fail-closed invalidity, especially disconnected/absent join
anchors, rather than JSON parsing (100% on all three). These values must never be cited as full-set
scores.

The local official HotpotQA fullwiki development contexts contain ten retrieved documents each,
but only 2,089/7,405 questions contain every gold document; gold-document recall is 55.71%. This is
reserved as a separate missing-evidence stress test and must not be mixed into the distractor table.

## Acceptance interpretation

VLDB does not require beating the absolute leaderboard under unrelated model/data settings. A
benchmark-centric submission should beat the strongest truly comparable 7B trained baseline, but a
systems paper can be competitive rather than numerically first if it establishes a material and
well-measured Pareto gain in correctness, fail-closed coverage, robustness, update maintenance,
latency, or cost. VeriJoin now has complete three-dataset results and inference/training/update cost
measurements. It still needs same-backbone answer-only and citation-only baselines, typed/query/equi
ablations, counterfactual/disconnected-evidence tests, a real versioned-source update workload, and
direct BLIP/GroundedCache comparisons before the systems claim is acceptance-ready. Full local
numbers and their non-SOTA status are in `docs/full_results.md`.
