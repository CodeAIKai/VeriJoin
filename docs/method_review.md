# Final method and acceptance audit (2026-08-31)

## Decision

Internal simulated review: **Accept, 7/10**. This is not a guarantee of external VLDB acceptance.
The defensible contribution is certified execution lineage and selective maintenance for learned
text joins, not multi-hop QA SOTA.

## Falsification outcomes

- **Universal QA improvement: rejected.** Under equal N=4 controls, VeriJoin minus free-literal
  Strict F1 is -2.55/+4.55/-1.46; the MuSiQue interval crosses zero.
- **Useful executable coverage: supported.** Certified coverage is 84.79/91.81/69.51%; literal
  programs certify zero results.
- **Typed reducer benefit: supported conditionally.** 2Wiki gains +4.55 Strict F1 over free literal,
  consistent with its deterministic comparison operators.
- **Fine-grained safe routing: supported for the declared contract.** Five update classes achieve
  100% declared-action conformance over every eligible certified output; structure changes fail
  closed to recompilation.
- **Natural-history selectivity: supported in a bounded workload.** 315 actual-program-linked
  Wikipedia revision events route to 303 reuse/10 replay/2 recompile, avoiding 99.37% of model calls
  relative to page invalidation.
- **Recompile usefulness: supported conditionally.** On 600 hash-selected numeric/date fact changes,
  stale-cache EM and old-plan validity are zero; recompile reaches 97.62-99.12 Strict F1 and
  99.5-100% recertification.

## Why the database contribution survives neighboring work

Executable LLM plans, provenance, and safe cache routing are not individually new. VeriJoin's narrow
intersection is that an untrusted compiler emits a restricted value-producing plan; a trusted VM
produces exact source-cell lineage during normal execution; and the same certificate controls
reuse/replay/recompile after updates. Restricting neural value generation is what makes deterministic
replay follow from unchanged versioned cells.

## Remaining scientific limits

One training seed, provided-context public dev results, limited MuSiQue composition coverage, a
50-title HotpotQA historical sample, conditional numeric/date recovery, and a labeled BLIP proxy.
These must remain visible. No artifact should describe the method as 7B or overall QA SOTA.

## Reproduction gates

- 22,398/22,398 examples; no skip or truncation.
- 89,592 candidates per N=4 protocol; 358,368 across four output protocols.
- 600/600 recovery cases and 4,800 recovery candidates.
- 65 tests pass; Ruff passes over `src`, `tests`, and `scripts`.
- Main PDF: 12 manuscript pages followed by one references-only page.
- Appendix PDF: 7 pages and self-contained as supplementary material.

A stable public artifact URL and author-approved license remain administrative requirements before
submission; they are intentionally not fabricated here.
