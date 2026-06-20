#!/usr/bin/env bash
# Phase 2 CPU efficiency sweep: each (model, method) in a FRESH process (clean RAM).
# Usage: phase2_matrix.sh "<models>" "<methods>" <n_train> <max_len> <threads>
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/hf_cache}
PY=.venv/bin/python

MODELS=${1:-"Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B"}
METHODS=${2:-"frozen lora loca_f mezo"}
NT=${3:-256}
ML=${4:-256}
THREADS=${5:-96}
ST=outputs/phase2_status.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ST"; }

mkdir -p outputs
log "phase2: models=[$MODELS] methods=[$METHODS] n_train=$NT max_len=$ML threads=$THREADS"
for model in $MODELS; do
  for method in $METHODS; do
    log "phase2: $model / $method (fresh process)"
    $PY scripts/phase2_efficiency.py --model "$model" --method "$method" \
        --n-train $NT --max-len $ML --batch-size 8 --threads $THREADS \
        2>&1 | grep -viE "warning|hf_token|loading weights|deprecat|it/s\]" | sed 's/^/    /'
  done
done
log "phase2: ALL DONE"
