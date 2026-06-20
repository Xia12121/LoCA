#!/usr/bin/env bash
# Phase 2 "memory ceiling" demo (C3): on a RAM-limited box, show where each method
# OOMs vs survives as model size grows. Each config is a fresh subprocess; if it is
# OOM-killed (exit 137 / no output), we record an OOM marker row.
# Usage: phase2_ceiling.sh "<models>" "<methods>" <dtype> <n_train> <max_len> <threads>
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/hf_cache}
PY=.venv/bin/python

MODELS=${1:-"Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B"}
METHODS=${2:-"lora loca_f"}
DTYPE=${3:-float32}
NT=${4:-32}; ML=${5:-128}; THREADS=${6:-16}
CEIL=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "?")
ST=outputs/ceiling_status.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ST"; }

log "ceiling: cgroup_limit=$CEIL bytes  models=[$MODELS] methods=[$METHODS] dtype=$DTYPE"
for model in $MODELS; do
  for method in $METHODS; do
    log "ceiling: $model / $method / $DTYPE"
    $PY scripts/phase2_efficiency.py --model "$model" --method "$method" --dtype "$DTYPE" \
        --n-train $NT --max-len $ML --batch-size 4 --threads $THREADS \
        > outputs/ceil_${method}_${DTYPE}.tmp 2>&1
    rc=$?
    grep -E "phase2\]|wall_per|base_ram|marginal" outputs/ceil_${method}_${DTYPE}.tmp | sed 's/^/    /'
    if [ $rc -ne 0 ]; then
      # 137 = SIGKILL (OOM killer); anything nonzero without a result row = failure
      reason="exit$rc"; [ $rc -eq 137 ] && reason="OOM"
      log "ceiling: $model / $method -> FAILED ($reason)"
      $PY - "$model" "$method" "$DTYPE" "$reason" <<'PY'
import sys; sys.path.insert(0,".")
from src.utils.logging_csv import ResultRow, append_row
model,method,dtype,reason=sys.argv[1:5]
append_row("outputs/efficiency.csv", ResultRow(method=method, model=model, task="cpu_efficiency",
    seed=0, metric="status", value=0.0, extra={"status":reason,"dtype":dtype}))
PY
    fi
  done
done
log "ceiling: DONE"
