# Final full-set experimental record (2026-08-31)

## Scope and protocol

All benchmark values are local, labeled, provided-context development results, not blind-test or
open-corpus scores. The evaluation covers every HotpotQA distractor-dev question (7,405), every
2WikiMultiHopQA dev question (12,576), and every answerable MuSiQue dev question (2,417): 22,398
questions with no sampling, skipped overlength inputs, or context truncation.

The main compiler and the answer-only, citation-only, and free-literal controls use Qwen2.5-7B,
independent QLoRA adapters, the same source examples, three training stages, 12,000 cumulative
exposures, and N=4 label-free selection. Each protocol therefore generates 89,592 candidates; the
four protocols total 358,368 candidates. The selector can use serving-time validity, answer
consensus, and mean generated-token likelihood, never reference labels.

## Main compiler results

Strict F1 assigns zero to malformed, ungrounded, or proof-invalid programs. Certified coverage is
the fraction executed by deterministic `copy/argmin/argmax/equal/common` operators.

| Dataset | Decode | Answer F1 | Strict F1 | Valid % | Evidence F1 | Certified coverage |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA | N=1 | 75.08 | 67.14 | 85.05 | 84.83 | 79.00 |
| HotpotQA | N=4 | 76.24 | 71.27 | 90.83 | 84.91 | 84.79 |
| 2Wiki | N=1 | 81.95 | 77.96 | 92.21 | 94.69 | 89.38 |
| 2Wiki | N=4 | 82.86 | 80.34 | 95.32 | 94.90 | 91.81 |
| MuSiQue | N=1 | 59.77 | 40.33 | 58.21 | 85.35 (document) | 58.17 |
| MuSiQue | N=4 | 61.51 | 46.68 | 69.55 | 86.07 (document) | 69.51 |

N=4 raises Answer F1 by 1.16/0.91/1.74 and Strict F1 by 4.12/2.38/6.34.
Paired Answer-F1 95% intervals are [0.78,1.56], [0.66,1.16], and [0.98,2.54].
The N=4 run generates 89,592 candidates in 5,538.20 seconds on one RTX 5090.

## Equal-budget output protocols

| Dataset | Protocol | Answer F1 | Strict F1 | Valid type/rate | Evidence F1 | Certified coverage |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA | Answer-only | 78.96 | — | parse 99.99 | — | — |
| HotpotQA | Citation-only | 79.32 | — | citation 100.00 | 85.08 | — |
| HotpotQA | Free literal | 78.73 | 73.82 | proof 92.95 | 84.51 | 0.00 |
| HotpotQA | VeriJoin | 76.24 | 71.27 | proof 90.83 | 84.91 | 84.79 |
| 2Wiki | Answer-only | 79.08 | — | parse 100.00 | — | — |
| 2Wiki | Citation-only | 80.02 | — | citation 99.99 | 94.82 | — |
| 2Wiki | Free literal | 78.67 | 75.79 | proof 95.56 | 94.53 | 0.00 |
| 2Wiki | VeriJoin | 82.86 | 80.34 | proof 95.32 | 94.90 | 91.81 |
| MuSiQue | Answer-only | 65.80 | — | parse 100.00 | — | — |
| MuSiQue | Citation-only | 64.21 | — | citation 100.00 | 38.11 | — |
| MuSiQue | Free literal | 63.79 | 48.14 | proof 71.25 | 85.61 | 0.00 |
| MuSiQue | VeriJoin | 61.51 | 46.68 | proof 69.55 | 86.07 | 69.51 |

VeriJoin minus answer-only Answer F1 is -2.72 (95% CI [-3.44,-1.99]), +3.78
([3.21,4.38]), and -4.29 ([-5.90,-2.68]). VeriJoin minus free-literal Strict F1 is
-2.55 ([-3.35,-1.75]), +4.55 ([3.97,5.15]), and -1.46 ([-3.07,0.19]); the
MuSiQue difference is not significant (p=0.087). This falsifies a universal accuracy benefit.
The surviving claim is a dataset-dependent quality/maintainability frontier: literal programs have
zero deterministic certificate coverage, while VeriJoin certifies 84.79/91.81/69.51%.

Near-perfect citation format is also insufficient: citation-only MuSiQue evidence F1 is 38.11,
versus 86.07 for the typed program. 2Wiki is the positive value-execution case because deterministic
comparison reducers are common; VeriJoin exceeds free literal by 4.55 Strict F1.

## Update routing and structural safety

On all N=4 valid outputs, the uniform-cell expected recomputation rate is 7.11/10.20/4.53%, a
92.89/89.80/95.47% reduction relative to invalidating each full displayed context. Mean snapshot
sizes after adding an ordered document/sentence-count digest are 870/926/958 bytes; bind medians are
165/147/231 us and verification medians 13.4/14.0/17.7 us.

Every eligible certified program receives five deterministic update classes:

- unread-cell content edit -> reuse;
- read-sentence rewrite retaining the operand -> replay;
- answer-value replacement -> recompile;
- sentence insertion -> recompile;
- sentence deletion -> recompile.

The declared action is taken in 100% of 6,278-6,279 HotpotQA, 11,546 2Wiki, and 1,680 MuSiQue
eligible cases per class. This establishes controller conformance, not new-plan accuracy.

## Actual-program Wikipedia history

Fifty HotpotQA titles are selected by SHA-256 order from titles cited by certified predictions. The
last revision at or before the official 2017-10-01 snapshot anchor and up to ten later revisions are
fetched. Forty-seven pages resolve; 34 contribute 36 actual program/page bindings. An event is
eligible only when the generated program's real evidence sentences and pointer quotes bind in the
old revision. Across 315 changed adjacent-revision events, routing is 303 reuse, 10 replay, and 2
recompile, avoiding 99.37% of 7B calls relative to page-version invalidation. This is not a complete
historical QA split.

## Controlled fact-change recovery

Recovery is conditioned on an originally correct, certified `copy` program with a numeric/date
answer occurring once in the full context. Eligible pools contain 464/523/207 cases; SHA-256 order
selects 200 per dataset. A deterministic +/-1 source edit makes the old plan invalid, and an oracle
repair is certified only as a workload sanity check. Oracle programs are never supplied to models.

| Dataset | Pool/used | Stale EM | Old-plan valid | VeriJoin Strict F1 | Recertified | Answer-only F1 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 464/200 | 0.00 | 0.00 | 97.80 | 99.5 | 94.12 |
| 2Wiki | 523/200 | 0.00 | 0.00 | 99.12 | 100.0 | 95.19 |
| MuSiQue | 207/200 | 0.00 | 0.00 | 97.62 | 99.5 | 91.32 |

Paired VeriJoin-minus-answer-only F1 is +3.68 (95% CI [0.77,6.62]), +3.92
([1.69,6.50]), and +6.30 ([2.22,10.53]). Because eligibility is defined by prior VeriJoin success,
these are conditional recovery results, not unconditional model rankings. Program and answer-only
generation take 237.2 and 110.8 seconds for 2,400 candidates each in one loaded base-model process.

## Provenance/cache comparison

On certified outputs, VeriJoin stores 888/934/958 bytes and verifies in 14.77/15.42/19.14 us.
The explicitly labeled BLIP-style deletion proxy stores 409/401/524 bytes but needs 52.55/42.44/95.00
additional deterministic subset executions; it is not an official BLIP reproduction. A
GroundedCache-style document-version gate has zero unsafe hits on relevant changes but falsely
invalidates 100% of results when an unread sentence in the same cited document changes; VeriJoin's
cell dependency has zero false invalidations on that workload.

## Reproducibility map

- Main N=4: `artifacts/results/stage3-*-full-fixed-execution-n4.json`
- Output protocols: `artifacts/results/protocol-{answer,citation,free-literal}-*-n4.json`
- Equal-budget pairing: `artifacts/results/paired-n4-*.json`
- Structural/update stress: `artifacts/results/update-{structural,stress}-*.json`
- Actual history: `artifacts/results/hotpotqa-2017-program-history.json`
- Recovery: `artifacts/results/recompile-recovery-{600,paired}.json`
- Provenance: `artifacts/results/provenance-structural-*.json`
- Per-example recovery predictions: `artifacts/predictions/update-recovery/*.jsonl`

The code passes 65 tests and Ruff. Results establish a systems Pareto contribution, not 7B or overall
QA SOTA. Remaining submission administration is a public archival artifact URL and an author-approved
license; neither is fabricated in the local package.
