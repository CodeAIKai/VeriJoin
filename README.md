# VeriJoin Artifact

This directory is the lightweight submission artifact for **VeriJoin: Executable Lineage for Selective Maintenance of Learned Text Joins**. It contains the implementation, configurations, tests, compact reports, and full per-example predictions used by the final paper tables. It deliberately excludes third-party benchmark corpora, the Qwen2.5-7B-Instruct base model, trained checkpoints, and virtual environments.

## What is included

- `src/verijoin/`: dataset adapters, typed IR, VM, compiler/evaluator, exact lineage, update controller, training and inference code.
- `tests/`: 50 unit and integration tests for parsing, execution, maintenance, evaluation and statistical routines.
- `configs/`: all staged QLoRA and full-split inference configurations.
- `scripts/`: orchestration scripts for the uncapped N=1/N=4 development runs.
- `artifacts/results/`: lightweight final reports supporting the paper tables.
- `artifacts/predictions/`: full final N=1/N=4, selector, and same-backbone per-example predictions plus run manifests.
- `docs/`: correctness contract, complete results, literature and internal method audit.
- `README_ORIGINAL.md`: the working-server README, retained for command-level detail. Its absolute paths are examples from the authors' machine and must be replaced locally.

All tracked files are source code or text artifacts. For this English-only public release, four
repeated non-English name aliases in saved quote fields were normalized to their English
romanization; the predictions, scores, and execution decisions were otherwise unchanged.

## Scope and data contract

The reported local evaluation covers every labeled official development example:

| Dataset | Split | Questions |
|---|---|---:|
| HotpotQA distractor | validation | 7,405 |
| 2WikiMultiHopQA | development | 12,576 |
| MuSiQue answerable | development | 2,417 |
| **Total** | | **22,398** |

The artifact does not redistribute these datasets. Obtain them from their official distributions and place them under one local root, referred to below as `$VERIJOIN_DATA_ROOT`. Dataset licenses and terms remain those of the original publishers.

## Environment

Recommended: Linux, Python 3.10+, CUDA-capable PyTorch for training/inference, and enough storage for the external model and benchmark files. The deterministic VM, reports and tests run without a GPU.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

The reference training run used one 32 GB RTX 5090, Qwen2.5-7B-Instruct, NF4 QLoRA rank 64, BF16 compute, response-only loss and gradient checkpointing. The paper reports observed hardware/runtime rather than claiming hardware invariance.

## Minimal deterministic checks

```bash
export VERIJOIN_DATA_ROOT=/absolute/path/to/datasets
.venv/bin/verijoin count --raw-root "$VERIJOIN_DATA_ROOT" --split dev
.venv/bin/verijoin audit --raw-root "$VERIJOIN_DATA_ROOT" --split dev
.venv/bin/pytest -q
```

The test suite checks fail-closed binding, typed reducers, query/evidence joins, cell-version lineage, refresh transitions, attacks, metrics and paired significance. It does not download external data.

## Training and uncapped evaluation

The staged training configurations are `configs/stage1-2k-qlora.json`, `configs/stage2-6k-qlora.json`, and `configs/stage3-12k-qlora.json`. The final uncapped inference configurations are `configs/stage3-infer-full.json` and `configs/stage3-guided-full.json`.

Before running, replace machine-specific `model`, `adapter`, input and output paths in the JSON configurations. Then follow `README_ORIGINAL.md` and:

```bash
PYTHONPATH=src .venv/bin/python -m verijoin.cli train --config configs/stage3-12k-qlora.json
bash scripts/run_full_dev.sh
bash scripts/run_fixed_n4_dev.sh
```

These are expensive commands. They require the externally obtained datasets, base model, constructed SFT files and the selected adapter. Candidate selection never reads gold answers or supporting facts.

## Paper-to-artifact map

| Paper result | Included report pattern |
|---|---|
| Completeness and full N=4 results | `stage3-*-full-fixed-execution-n4.json` |
| N=1 contract and N=4 contract | `missing-contract-n1-*.json`, `missing-contract-*.json` |
| Same-backbone answer/citation/free-literal baselines | `missing-answer-model-*.json`, `missing-citation-model-*.json`, `missing-free-literal-model-*.json` |
| Contract ladder and diagnostics | `missing-contract-*.json`, `missing-operators-*.json` |
| Post-generation attacks | `missing-attacks-*.json` |
| Synthetic update maintenance | `missing-updates-n4-*.json` |
| Real Wikipedia revisions | `missing-wikipedia-temporal-certified-150x5.json` |
| Provenance/cache comparison | `missing-provenance-certified-full-*.json` |
| Paired significance | `missing-paired-*.json`, `missing-significance-n4-vs-n1-*.json` |
| Candidate selector analyses | `stage3-*-full-*.json` |

The compact reports contain aggregates; `artifacts/predictions` contains the corresponding final per-example JSONL files and manifests. Obsolete 200-example, wrong-prompt, and pre-fix engineering runs are excluded. Checkpoints/adapters should be deposited as a separate versioned release with checksums when redistribution is permitted.

## Omitted files and why

- Base model and tokenizer: externally licensed Qwen2.5-7B-Instruct distribution.
- Benchmark raw data: governed by the original dataset terms.
- Checkpoints/adapters and optimizer states: tens of GB; archive separately if redistribution is permitted.
- Virtual environments and caches: machine-specific and reproducible from `pyproject.toml`.
- Obsolete 200-example or known-wrong-prompt diagnostics: engineering history, not evidence for paper claims.

## Submission checklist

1. Select and add an explicit open-source license approved by all authors.
2. Replace all absolute local paths and test in a clean environment.
3. Deposit this code-and-predictions package plus permitted model artifacts in a stable public archival repository.
4. Record repository URL, release tag/DOI and SHA-256 checksums in the paper and submission form.
5. PVLDB 2027 is single-blind: keep manuscript author names, while removing unrelated local user names, hostnames, secrets and private URLs from the public artifact.
6. Re-run the 50 tests and static checks from this packaged directory.

No license is chosen automatically here because that is an author/legal decision.
