# Full development-set results (2026-08-25)

## Evaluation contract and completeness

These are local, labeled development-set results, not blind-test submissions. The run covers every
example in HotpotQA distractor dev (7,405), 2WikiMultiHopQA dev (12,576), and MuSiQue-answerable
dev (2,417): 22,398 questions in total. The fixed N=4 run generated four candidates for every
question (89,592 candidates), with zero skipped overlength examples and no context truncation.

All rows use one Qwen2.5-7B-Instruct compiler with the same QLoRA adapter and VM. The main N=4
selector is fixed across datasets and does not read labels: strongest executable tier, then answer
consensus, then true mean generated-token log-likelihood. A previously generated N=4 diagnostic
lacked real token log-probabilities and is excluded from all claims; only files containing `fixed`
are used below.

## Main results

Answer EM/F1 follows the benchmark answer normalization. Strict F1 assigns zero to malformed,
ungrounded, or disconnected programs. Certified coverage is the fraction executed by deterministic
`copy/argmin/argmax/equal/common` operators; certified conditional F1 is computed only on that
subset and is not an unconditional accuracy score.

| Dataset | Decode | Examples | Answer EM | Answer F1 | Strict F1 | Valid % | Evidence F1 | Joint F1 | Certified coverage | Certified conditional F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | greedy N=1 | 7,405 | 61.82 | 75.08 | 67.14 | 85.05 | 84.83 | 65.72 | 79.00 | 78.51 |
| HotpotQA | **execution N=4** | 7,405 | **62.73** | **76.24** | **71.27** | **90.83** | **84.91** | **66.75** | **84.79** | 78.07 |
| 2Wiki | greedy N=1 | 12,576 | 75.91 | 81.95 | 77.96 | 92.21 | 94.69 | 80.46 | 89.38 | 84.46 |
| 2Wiki | **execution N=4** | 12,576 | **76.93** | **82.86** | **80.34** | **95.32** | **94.90** | **81.42** | **91.81** | 84.42 |
| MuSiQue | greedy N=1 | 2,417 | 50.02 | 59.77 | 40.33 | 58.21 | 85.35 (document) | n/a | 58.17 | 69.33 |
| MuSiQue | **execution N=4** | 2,417 | **51.55** | **61.51** | **46.68** | **69.55** | **86.07 (document)** | n/a | **69.51** | 67.15 |

N=4 improves unconditional Answer F1 over N=1 by +1.16, +0.91, and +1.74 points, and strict F1
by +4.12, +2.38, and +6.34 points on HotpotQA, 2Wiki, and MuSiQue respectively. The larger strict
gains show that execution guidance mainly reduces invalid/disconnected outputs. The reduction in
certified conditional F1 on HotpotQA and MuSiQue is a coverage-composition effect and is reported
rather than hidden.

## Candidate-selector ablation

All N=4 selectors see exactly the same four generated candidates. The BGE pairwise ranker and small
answer-group meta-ranker are trained only from training-derived questions; they never see dev
labels. The meta-ranker is 0.08 F1 above execution selection on HotpotQA but loses on the other two
datasets and lowers strict F1 on all three. The single fixed execution selector is therefore the
pre-declared main method rather than choosing a selector per dataset.

| Selector over N=4 | Hotpot Answer/Strict F1 | 2Wiki Answer/Strict F1 | MuSiQue Answer/Strict F1 |
|---|---:|---:|---:|
| true likelihood | 75.15 / 67.80 | 82.45 / 78.96 | 60.38 / 42.20 |
| BGE pairwise ranker | 75.75 / 70.38 | 82.75 / 80.11 | 60.74 / 45.32 |
| answer-group meta-ranker | 76.33 / 71.14 | 82.69 / 80.13 | 60.36 / 45.18 |
| **VM execution + consensus + likelihood** | **76.24 / 71.27** | **82.86 / 80.34** | **61.51 / 46.68** |

The ranker result is a useful negative experiment: train-derived holdout accuracy did not transfer
into a consistently better full-dev selector. It must not replace the main method post hoc.

## Run completeness and cost

| Dataset | Questions | Candidates | Generation seconds | Questions/s | Input tokens | Output tokens | Skipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 7,405 | 29,620 | 1,859.77 | 3.98 | 13,582,289 | 3,860,745 | 0 |
| 2Wiki | 12,576 | 50,304 | 2,607.20 | 4.82 | 17,711,614 | 7,279,250 | 0 |
| MuSiQue | 2,417 | 9,668 | 1,071.23 | 2.26 | 7,781,684 | 1,684,433 | 0 |

Total N=4 generation time was 5,538.20 seconds (92.30 minutes) on one RTX 5090. Training used
NF4 QLoRA rank 64 and 12,000 cumulative balanced sample exposures. The final 6,000-exposure stage
took 5,135.76 seconds, peaked at 23.57 GB allocated / 32.37 GB reserved, and selected its checkpoint
using only a training-derived holdout. The filtered SFT files contain 267,364 train rows and 5,437
holdout rows, with zero ID overlap with each other or with dev.

## Versioned-lineage update experiment

The current update test mutates one referenced cell and one unreferenced distractor cell for every
valid N=1 program. It validates implementation semantics, but it is synthetic and is not yet a
real time-versioned corpus benchmark.

| Dataset | Valid programs | Relevant detected | Irrelevant preserved | Selective recompute | Recompute reduction | Snapshot bytes | Bind p50/p95 us | Verify p50/p95 us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 6,298 | 100% | 100% | 7.09% | 92.91% | 781.66 | 153.30 / 256.49 | 5.44 / 7.56 |
| 2Wiki | 11,596 | 100% | 100% | 10.26% | 89.74% | 840.29 | 132.07 / 427.40 | 5.91 / 7.81 |
| MuSiQue | 1,407 | 100% | 100% | 4.51% | 95.49% | 867.63 | 215.16 / 496.12 | 6.49 / 9.56 |

## Claim status

These results establish a complete, reproducible local 7B baseline and a consistent N=4 gain. They
do **not** establish 7B SOTA: there is no authoritative leaderboard filtered by model size and
training recipe, and nearby 7B papers use incompatible retrieval modes, metrics, or sampled
evaluation. They also do not establish blind-test SOTA because no official test submission has been
made. Any paper table must keep official blind-test systems, same-scale 7B systems, open-corpus
systems, and these closed-context dev results in separate blocks.

Before an acceptance-oriented submission, the lineage contribution still needs a real source-update
workload, adversarial disconnected/counterfactual tests, same-backbone answer-only and citation-only
ablations, and direct provenance/cache comparisons with BLIP and GroundedCache.

## Reproducibility artifacts

- Main reports: `artifacts/results/stage3-*-full-fixed-execution-n4.json`
- Greedy reports: `artifacts/results/stage3-*-full-greedy.json`
- Update reports: `artifacts/results/stage3-*-full-greedy-updates.json`
- Full candidate files and manifests: `artifacts/predictions/stage3-*-full-fixed-n4.jsonl*`
- Fixed full-run script: `scripts/run_fixed_n4_dev.sh`
- Final adapter manifest: `artifacts/checkpoints/stage3-12k/run_manifest.json`
- Ranker manifest: `artifacts/checkpoints/ranker-bge-lora/training_manifest.json`
