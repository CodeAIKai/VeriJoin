# Reproducibility package contents

This directory contains the final VeriJoin source, tests, configurations, scripts, documentation,
aggregate reports, complete per-example benchmark predictions, controlled-recovery predictions, and
the 50-title actual-program Wikipedia revision cache used by the paper.

Not included:

- Qwen2.5-7B base weights;
- the four 2.5GB QLoRA adapter directories;
- raw HotpotQA/2Wiki/MuSiQue datasets;
- Python virtual environments;
- an author-selected license.

Paths in packaged configuration and shell files use `/path/to/...` placeholders. Set them to local
model, adapter, dataset, and project paths. The exact library environment is in
`docs/environment.md`; final metrics and file mappings are in `docs/full_results.md`.

The local copy deliberately has no fabricated public artifact URL and no guessed license. Authors
must publish a stable archive, choose a license, and update the availability metadata before
submission.
