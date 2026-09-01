# Reproducibility environment

The reported runs used one NVIDIA RTX 5090 (32 GB), CUDA 13.0, Python 3.10, and the following
inference packages: vLLM 0.19.1+cu130, PyTorch 2.10.0+cu130, Transformers 4.57.6, PEFT 0.19.1,
NumPy 2.2.6, and PyArrow 24.0.0. The lightweight CPU evaluation environment is described by
`pyproject.toml`; the CUDA/vLLM environment must use wheels compatible with the host driver.

The base model is Qwen2.5-7B-Instruct. It, the three benchmark datasets, and trained checkpoints are
not redistributed in the lightweight artifact because their licenses and sizes differ. Paths are
configured explicitly in JSON files under `configs/`. Full prediction JSONL files, task-aware N=4
selection metadata, aggregate reports, and MediaWiki revision IDs/SHA-1 values are retained.

Core validation:

```bash
PYTHONPATH=src ./.venv/bin/ruff check src tests scripts
PYTHONPATH=src ./.venv/bin/pytest -q
```

Full same-backbone protocol evaluation:

```bash
./scripts/run_protocol_n4.sh
```

Program-linked historical evaluation and controlled factual-update recovery are implemented by
`evaluate-hotpot-history` and `scripts/run_recompile_recovery.py`, respectively. Neither command
reads benchmark labels during candidate selection. See `docs/full_results.md` for artifact-to-claim
mapping and exact result filenames.
