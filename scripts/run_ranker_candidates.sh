#!/usr/bin/env bash
set -euo pipefail

project_root="/path/to/verijoin"
raw_root="/path/to/datasets"
config="${1:-configs/stage3-guided-train-ranker.json}"

cd "$project_root"
export PYTHONPATH="$project_root/src"

for dataset in hotpotqa 2wiki musique; do
  output="artifacts/ranker_candidates/stage3-${dataset}-train-n4.jsonl"
  ./.venv-infer/bin/python -m verijoin.cli infer \
    --config "$config" --raw-root "$raw_root" --dataset "$dataset" --split train \
    --output "$output"
done
