#!/usr/bin/env bash
# Phase 1 full matrix for one model: tasks x methods x seeds -> outputs/results.csv
# Usage: phase1_matrix.sh <model> <eta> <T> "<tasks>" "<methods>" "<seeds>"
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-Qwen/Qwen2.5-0.5B}
ETAS=${2:-"0.005 0.002"}
T=${3:-15}
TASKS=${4:-"sst2 format_json alpaca"}
METHODS=${5:-"frozen lora loca_f"}
SEEDS=${6:-"0 1 2"}
PY=${PY:-/venv/main/bin/python}
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

echo "[matrix] model=$MODEL etas=[$ETAS] T=$T tasks=[$TASKS] methods=[$METHODS] seeds=[$SEEDS]"
for task in $TASKS; do
  ml=512; [ "$task" = "sst2" ] && ml=128; [ "$task" = "format_json" ] && ml=128
  nt=3000; [ "$task" = "alpaca" ] && nt=4000
  echo "[matrix] === $MODEL / $task (max_len=$ml n_train=$nt) ==="
  $PY scripts/phase1.py --model "$MODEL" --task "$task" \
      --methods $METHODS --seeds $SEEDS \
      --n-train $nt --n-eval 800 --max-len $ml --batch-size 16 \
      --device cuda --loca-etas $ETAS --T $T --feedback sketch 2>&1 \
    | grep -viE "warning|hf_token|deprecat|loading weights|generating|examples/s|it/s\]"
done
echo "[matrix] DONE model=$MODEL"
