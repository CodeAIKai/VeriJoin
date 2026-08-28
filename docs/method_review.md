# Method decision record (2026-08-25)

## Rejected candidates

1. Adaptive retrieval depth, fewer LLM calls, or a beam over passage chains is not sufficient:
   EfficientRAG, Beam Retrieval, FrugalRAG, and Interact-RAG already cover these contributions.
2. A generic semantic-operator DAG with cost-based replanning is not sufficient: CAESURA, AOP,
   LOTUS, DocETL, Abacus, and Sema make this space crowded.
3. End-to-end conformal calibration is not sufficient: TRAQ and ConRAD already provide statistical
   guarantees for RAG or multi-hop neural queries.
4. Post-hoc evidence scoring, free-form faithful traces, or LLM-simulated logic are not sufficient:
   PAVE, CRAFT, and FLARE are direct recent neighbors. Source-version cache validation alone is also
   insufficient because GroundedCache already studies that problem.
5. Executable multi-hop programs and compiler feedback are not sufficient after PyRAG (May 2026),
   and verifiable provenance alone is not sufficient after BLIP (VLDB 2026).

## Selected hypothesis

VeriJoin introduces a lineage-constrained execution model for learned queries over text. The model
emits exact source quotes rather than brittle numeric offsets; the VM late-binds them to the current
input. An `equi` join consumes cited evidence and proves that the normalized value occurs in its
other document. A `query` join binds a literal question constant (`doc=sent=-1`) to a cited document.
The VM requires the resulting proof graph to be connected for bridge/composition programs.

`argmin`, `argmax`, `equal`, and `common` compute the answer inside the VM from grounded operands.
`copy` is restricted to cited evidence. The remaining `bool` form is deliberately reported as a
learned semantic predicate over cited evidence, not misrepresented as deterministic deduction.
Malformed programs, absent quotes, disconnected lineage, and literals fail closed.

The evaluator therefore separates ordinary benchmark Answer F1, fail-closed strict F1, and
lineage-certified coverage/conditional F1. “Certified” has a deliberately narrow meaning: the result
is a deterministic function of versioned, byte-exact input cells. It is not a claim that the model's
chosen evidence is semantically relevant or that the benchmark answer is correct.

Successful plans can be bound to an exact source-version snapshot. The snapshot fingerprints every
question, title, and sentence cell read by evidence references, joins, and answer operators. A
cached answer is therefore selectively invalidated after a relevant corpus change without being
invalidated by an unreferenced distractor update. This is execution-time state maintained by the
system, not an extra string the model must learn to emit.

This is materially different from asking an LLM to produce JSON-formatted reasoning. JSON syntax
is only the wire format; the VM gives the program operational semantics and rejects values that do
not have source lineage.

At inference time, an optional execution-guided decoder can sample several programs from the same
7B compiler. It first prefers the strongest executable tier, then answer consensus within that tier,
then mean token log-likelihood. It has no access to labels. This is a mechanism for exposing the
value of VM feedback, not a free accuracy gain: the paper must retain one-candidate greedy decoding,
report every generated token and wall-clock cost, and include likelihood-only candidate selection as
an ablation. Best-of-N or self-consistency by itself is not claimed as the database contribution.

## Conditions for a defensible VLDB submission

- A common IR/compiler must work across all three datasets without dataset-specific inference code.
- Full labeled development sets must be evaluated; blind-test results must not be inferred.
- Checkpoint selection uses only the deterministic train-derived holdout. The fixed N=1 and N=4
  decoding protocols must both run on the complete development sets; 200-example diagnostics cannot
  choose the reported winner.
- If the paper claims QA SOTA, the main result must beat strong same-scale trained readers under the
  same context/retrieval mode. A systems claim may instead establish a clear accuracy/coverage/update
  cost Pareto frontier, but it still needs a same-backbone answer-only baseline.
- Ablations must isolate span grounding, join validation, deterministic operators, and training data.
- Multi-candidate decoding must be compared with greedy and likelihood-only selection at the same N;
  full-set headline results must state N next to every score.
- The systems section must report throughput, p50/p95 latency, GPU memory, tokens, invalid-program
  rate, lineage storage/invalidation cost, and scaling with documents/hops.
- Open-corpus retrieval and closed distractor evaluation must be separate.
- Robustness must include counterfactual answer swaps and disconnected-evidence attacks; ordinary
  benchmark F1 alone cannot establish that the execution constraints matter.
- Query joins and deterministic reducers must be reported separately. Their oracle coverage cannot
  be presented as model accuracy.
- The answer-only/citation-only/typed-VM ablations must use the same Qwen2.5-7B checkpoint and data
  budget; otherwise the claimed gain is not attributable to execution semantics.

## Second self-review: the rejected v1 implementation

The first implementation only verified that a copied answer occurred in cited text and allowed a
finite yes/no literal. That was not sufficient: a reviewer could correctly describe it as structured
output wrapping because the model still selected the comparison answer. It was discarded before the
long training run. Version 2 adds executable typed reducers, exact query/evidence joins, and connected
lineage. The remaining semantic-boolean path is exposed as a limitation and an explicit ablation.

## Current falsification gates

1. If the 7B compiler cannot maintain high parse/valid rates on held-out examples, stop; the method
   is not ready for full evaluation.
2. If typed execution does not improve counterfactual consistency over citation-only generation,
   the database contribution is not empirically justified.
3. If accuracy drops below a same-scale answer-only SFT baseline at comparable compute, the paper
   needs a clear risk/coverage or robustness Pareto improvement, not an unsupported SOTA claim.
4. If full labeled development results are weak, do not infer hidden-test performance.

No method can be guaranteed acceptance before results and reviewer judgment. This direction is
selected because it has a falsifiable database contribution and is feasible on one 32GB GPU.

## Third self-review after PyRAG and BLIP

PyRAG invalidates any broad claim that VeriJoin is the first executable approach to multi-hop QA.
BLIP invalidates any broad claim that associating an LLM output with verifiable source subsets is
new. The surviving claim is narrower: one local 7B call compiles a closed text query to a safe typed
IR; the VM, rather than another LLM call, derives certified outputs from exact versioned cells and
supports source-cell invalidation. PyRAG's Python program invokes opaque neural `answer()` calls and
targets open retrieval; BLIP is a model-agnostic post-hoc reproduction procedure. These distinctions
are technically real, but they are not enough by assertion. Acceptance still requires full accuracy,
certified-coverage, update, robustness, and cost results to establish a useful Pareto frontier.

## Stage-one review (2k sampled examples; not a final result)

On the first 200 development examples per dataset, parsing is 100%. Strict validity is 72.5% on
HotpotQA, 93.5% on 2Wiki, and 62.5% on MuSiQue. Official-style Answer F1 is 68.09, 75.51, and 61.26;
the stricter fail-closed values are 53.95, 74.48, and 46.96; conditional on a valid program it is
74.41, 79.66, and 75.13. The evidence F1 values are already 82.57, 93.07, and 83.67. This passes the
gate for continued training because the remaining gap is concentrated in
learnable quote/join validity rather than a low oracle ceiling or parser collapse. It does not pass
the gate for a result claim: HotpotQA and MuSiQue remain below the same-scale target, and none of
the three numbers covers the complete development split.

The archived CRAFT arXiv v1 target reported 80.10 HotpotQA, 84.18 2Wiki, and 63.06 MuSiQue for its
best trace variants. It evaluated ten random 1K-example runs per dataset rather than the complete
development sets and used substantially more training/judge compute. Its current arXiv v3 was
substantially revised, so the old numbers must be explicitly version-pinned rather than described as
the current SOTA.

## Fourth self-review after complete development evaluation

The final fixed N=4 protocol covers all 22,398 labeled development questions and skips none. Answer
F1 is 76.24 HotpotQA, 82.86 2Wiki, and 61.51 MuSiQue, improving over the same checkpoint's greedy
N=1 values by 1.16, 0.91, and 1.74 points. Strict fail-closed F1 improves by 4.12, 2.38, and 6.34
points. Complete values and manifests are recorded in `docs/full_results.md`.

This passes the engineering feasibility and full-coverage gates: one 32GB RTX 5090 can train the
compiler and complete N=4 inference, the VM increases valid/certified coverage, and exact read-set
snapshots reduce synthetic recomputation by 89.74--95.49% while detecting every referenced-cell
mutation in the implemented test.

It does **not** pass a QA-SOTA gate. The full-dev F1 values are below the archived CRAFT v1 sampled
references on all three datasets, and official blind-test systems are higher under different model
classes and protocols. A BGE pairwise ranker and an answer-group meta-ranker were trained using only
training-derived examples; neither improved consistently on all three full dev sets. The fixed VM
execution selector remains the main result rather than selecting a post-hoc winner per dataset.

### Acceptance verdict

The idea is plausible as a VLDB systems paper only under the narrowed claim: **incremental view
maintenance and byte-exact data lineage for learned text joins**, not “a new multi-hop QA SOTA,”
“the first executable QA program,” or “the first verifiable LLM provenance.” PyRAG, BLIP, and
GroundedCache make those broader claims untenable.

The current artifact is not yet acceptance-ready, and no honest review can guarantee acceptance.
The following are blocking experiments rather than optional polish:

1. Build a real, time-versioned Wikipedia/source update workload and report stale-answer serving,
   invalidation precision/recall, recomputation, and answer changes. The current single-cell mutation
   test proves implementation behavior but is synthetic.
2. Compare directly with BLIP on provenance generation/replay size, calls, latency, and reproduction;
   compare with GroundedCache on unsafe cache hits, invalidations, and savings under the same update
   stream.
3. Add same-Qwen2.5-7B, same-data answer-only and citation-only controls, plus no-connectivity,
   free-literal, query/equi, deterministic-operator, and N=1/N=4 ablations.
4. Add counterfactual entity/number swaps, disconnected evidence, distractor poisoning, and missing
   source attacks. Demonstrate that fail-closed behavior buys robustness rather than merely lowering
   coverage.
5. State and prove the VM-level property: for the certified operator subset, a result is a
   deterministic function of its recorded read set and unchanged cells preserve replay. Keep this
   distinct from semantic correctness.

If those experiments show a meaningful Pareto frontier at similar Answer F1, the contribution is
well aligned with database interests in provenance, query execution, and incremental maintenance.
If they do not, further QA-only tuning should not be used to rescue the claim; the hypothesis has
been falsified and the paper direction should change.
