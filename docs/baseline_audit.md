# External baseline audit

## GroundedCache

- Repository: <https://github.com/syedhumarahim/grounded-cache-router>
- Pinned local commit: `2fbe16e09fee7994b8e0069e87086c912f2adc0d`
- Verification command: `PYTHONPATH=src python -m pytest -q`
- Result on 2026-08-31: `19 passed`

VeriJoin does not claim to reproduce GroundedCache's published answer-quality experiments. The paper
implements the four-gate exact-repeat/source-version semantics with one benchmark document as one
chunk, and compares only relevant-update unsafe hits and same-document false invalidation.

## BLIP

BLIP is cited as PVLDB 19(11):2992--3005, DOI `10.14778/3836663.3836668`. No matching public
implementation was located during the artifact audit. The reported comparison is therefore named a
one-pass deletion proxy with deterministic VM replay, not an official BLIP reproduction. Its replay
latency is a favorable lower bound because it avoids black-box model calls.
