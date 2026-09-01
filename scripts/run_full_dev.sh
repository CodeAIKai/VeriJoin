#!/usr/bin/env bash
set -euo pipefail

project_root="/path/to/verijoin"
raw_root="/path/to/datasets"
greedy_config="${1:-configs/stage3-infer-full.json}"
guided_config="${2:-configs/stage3-guided-full.json}"

cd "$project_root"
export PYTHONPATH="$project_root/src"

for dataset in hotpotqa 2wiki musique; do
  greedy="artifacts/predictions/stage3-${dataset}-full-greedy.jsonl"
  guided="artifacts/predictions/stage3-${dataset}-full-guided-n4.jsonl"
  likelihood="artifacts/predictions/stage3-${dataset}-full-likelihood-n4.jsonl"

  ./.venv-infer/bin/python -m verijoin.cli infer \
    --config "$greedy_config" --raw-root "$raw_root" --dataset "$dataset" --split dev \
    --output "$greedy"
  ./.venv/bin/python -m verijoin.cli evaluate \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --predictions "$greedy" \
    --report "artifacts/results/stage3-${dataset}-full-greedy.json"
  ./.venv/bin/python -m verijoin.cli evaluate-updates \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --predictions "$greedy" \
    --report "artifacts/results/stage3-${dataset}-full-greedy-updates.json"

  ./.venv-infer/bin/python -m verijoin.cli infer \
    --config "$guided_config" --raw-root "$raw_root" --dataset "$dataset" --split dev \
    --output "$guided"
  ./.venv/bin/python -m verijoin.cli evaluate \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --predictions "$guided" \
    --report "artifacts/results/stage3-${dataset}-full-guided-n4.json"

  ./.venv/bin/python -m verijoin.cli reselect \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --source "$guided" \
    --selection likelihood --output "$likelihood"
  ./.venv/bin/python -m verijoin.cli evaluate \
    --raw-root "$raw_root" --dataset "$dataset" --split dev --predictions "$likelihood" \
    --report "artifacts/results/stage3-${dataset}-full-likelihood-n4.json"
done
