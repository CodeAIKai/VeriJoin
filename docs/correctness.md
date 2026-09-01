# Correctness contract

This document states the narrow property implemented by VeriJoin. It does not equate provenance
with semantic correctness.

## Objects committed by a snapshot

A successful binding commits to:

1. the dataset and query identifier;
2. the VM semantics version;
3. the canonical typed program, excluding non-semantic metadata;
4. the exact answer bytes produced by that program;
5. a SHA-256 digest of ordered document identifiers and per-document sentence counts; and
6. SHA-256 versions of every question, title, or sentence cell read by evidence, joins, and answer
   operators.

`bind_lineage` first executes the program and rejects invalid plans. Equi-joins bind the particular
right-hand title or sentence that witnesses membership, rather than conservatively invalidating the
whole joined document. `verify_result_binding` detects a substituted plan, answer, or VM version;
`verify_lineage` detects a missing or bytewise-changed source cell and any insertion/deletion that
changes the candidate-context cell-ID domain.

## Lineage-certified result property

For a valid `copy`, `argmin`, `argmax`, `equal`, or `common` program, the answer is a deterministic
function of the committed program and committed source cells. Therefore, with the same VM semantics,
if the program/answer binding and structural digest verify and every source-cell digest remains
current, replay returns the same answer. A content edit to an unbound distractor cannot change that
replay; insertion, deletion, or resegmentation conservatively requests recompilation before indices
can shift.

The implementation checks the premises rather than trusting a model statement: quotes must occur in
their declared source, answer operands must be cited, joins need a concrete witness, and multi-document
bridge/composition graphs must be connected.

## Explicit non-guarantees

- Certification does not prove that the selected evidence is relevant to the natural-language
  question or that the answer matches a benchmark label.
- A valid learned `bool` result is not lineage-certified because the VM does not derive its truth value.
- Hash commitments detect updates; they do not authenticate a malicious storage provider without an
  external trust root.
- Collision resistance relies on SHA-256, and replay additionally relies on the declared VM semantics
  version matching the implementation.

These boundaries are reflected in separate Answer F1, strict F1, lineage-certified coverage, and
conditional certified F1 metrics.
