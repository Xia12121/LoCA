#!/usr/bin/env bash
# CPU-1 (32GB box, limited) — LIGHT three-way slice that complements CPU-2:
# 0.5B HellaSwag + ARC three-way (MeZO vs LoRA vs LoCA). threads=16 (CPU-1 optimum),
# 0.5B only (32GB can't do 3B/7B fp32). Own CSV cpu1_quality.csv. Resumable.
#
# Launch: cd /data/loca && HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data/hf_cache \
#         setsid bash scripts/pipeline_cpu1.sh > outputs/pipeline_cpu1.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/hf_cache}
export OMP_NUM_THREADS=16
PY=.venv/bin/python
QCSV=outputs/cpu1_quality.csv
ST=outputs/pipeline_cpu1_status.log
LOGS=outputs/logs; mkdir -p "$LOGS" outputs/snapshots; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name: SKIP"; return; fi
  local slog="$LOGS/cpu1_${name}.log"
  log "stage $name: START (detail -> $slog)"
  if "$@" >> "$slog" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name: FAILED (continuing)"; fi
}

threeway(){
  for task in hellaswag arc_easy; do
    $PY scripts/phase1.py --model Qwen/Qwen2.5-0.5B --task "$task" \
        --methods frozen lora mezo loca_f --seeds 0 \
        --n-train 2000 --n-eval 500 --max-len 256 --batch-size 8 \
        --device cpu --threads 16 --loca-etas 0.003 0.006 --T 12 --mezo-steps 3000 \
        --csv "$QCSV" || return 1
  done
}
stage q_05b_commonsense threeway
log "PIPELINE_CPU1_ALL_DONE"
