#!/usr/bin/env bash
set -euo pipefail

project_root="/path/to/verijoin"
raw_root="/path/to/datasets"
config="${1:-configs/stage3-guided-full.json}"
ranker_model="/path/to/bge-reranker-v2-m3"
ranker_adapter="artifacts/checkpoints/ranker-bge-lora"

cd "$project_root"
export PYTHONPATH="$project_root/src"

for dataset in hotpotqa 2wiki musique; do
  candidates="artifacts/predictions/stage3-${dataset}-full-fixed-n4.jsonl"
  likelihood="artifacts/predictions/stage3-${dataset}-full-fixed-likelihood-n4.jsonl"
  ranker="artifacts/predictions/stage3-${dataset}-full-fixed-ranker-n4.jsonl"

  ./.venv-infer/bin/python -m verijoin.cli infer \
    --config "$config" --raw-root "$raw_root" --dataset "$dataset" --split dev \
    --output "$candidates"
  ./.venv/bin/python -m verijoin.cli evaluate \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --predictions "$candidates" \
    --report "artifacts/results/stage3-${dataset}-full-fixed-execution-n4.json"

  ./.venv/bin/python -m verijoin.cli reselect \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --source "$candidates" \
    --selection likelihood --output "$likelihood"
  ./.venv/bin/python -m verijoin.cli evaluate \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --predictions "$likelihood" \
    --report "artifacts/results/stage3-${dataset}-full-fixed-likelihood-n4.json"

  ./.venv/bin/python -m verijoin.cli rank-candidates \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --source "$candidates" \
    --output "$ranker" --model "$ranker_model" --adapter "$ranker_adapter" \
    --batch-size 64 --max-length 512
  ./.venv/bin/python -m verijoin.cli evaluate \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --predictions "$ranker" \
    --report "artifacts/results/stage3-${dataset}-full-fixed-ranker-n4.json"
done
