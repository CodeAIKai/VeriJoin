#!/usr/bin/env bash
set -euo pipefail

project_root="/path/to/verijoin"
raw_root="/path/to/datasets"

cd "$project_root"
export PYTHONPATH="$project_root/src"

for task in answer citation free-literal; do
  config="configs/ablations/$task-n4.json"
  task_name="$task"
  for dataset in hotpotqa 2wiki musique; do
    predictions="artifacts/predictions/ablations/$task-$dataset-n4.jsonl"
    report="artifacts/results/protocol-$task-$dataset-n4.json"
    ./.venv-infer/bin/python -m verijoin.cli infer \
      --config "$config" --raw-root "$raw_root" --dataset "$dataset" --split dev \
      --output "$predictions"
    if [[ "$task" == "free-literal" ]]; then
      ./.venv/bin/python -m verijoin.cli evaluate \
        --raw-root "$raw_root" --dataset "$dataset" --split dev \
        --predictions "$predictions" --allow-literal --report "$report"
    else
      ./.venv/bin/python -m verijoin.cli evaluate-output-baseline \
        --raw-root "$raw_root" --dataset "$dataset" --split dev \
        --predictions "$predictions" --task "$task_name" --report "$report"
    fi
  done
done
