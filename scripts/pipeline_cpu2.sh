#!/usr/bin/env bash
# CPU-2 (64GB box) pipeline — two jobs the small 32GB box could not do:
#   (A) THREE-WAY quality comparison MeZO vs LoRA vs LoCA across several benchmarks
#       (the key deliverable) -> cpu2_quality.csv
#   (B) Large-model fp32 feasibility (3B/7B that OOM'd at 32GB) -> cpu2_efficiency.csv
# Resumable + per-stage logs + snapshots, server-side.
#
# Launch: cd /data/loca && HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data/hf_cache \
#         setsid bash scripts/pipeline_cpu2.sh > outputs/pipeline_cpu2.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/hf_cache}
export OMP_NUM_THREADS=${TH:-32}
PY=.venv/bin/python
TH=${TH:-32}
QCSV=outputs/cpu2_quality.csv
ECSV=outputs/cpu2_efficiency.csv
ST=outputs/pipeline_cpu2_status.log
SNAP=outputs/snapshots; LOGS=outputs/logs
mkdir -p "$SNAP" "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name: SKIP"; return; fi
  local slog="$LOGS/cpu2_${name}.log"
  log "stage $name: START (detail -> $slog)"
  echo "===== $(date +%F_%T) $name =====" >> "$slog"
  if "$@" >> "$slog" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name: FAILED (continuing)"; fi
  cp -f "$QCSV" "$SNAP/cpu2_quality_after_$name.csv" 2>/dev/null
}

# ---- (A) THREE-WAY quality: MeZO vs LoRA vs LoCA on several benchmarks ----- #
# All on CPU. MeZO is slow on CPU (this itself shows ZO is impractical here).
threeway(){  # threeway <model> <eta-list> <mezo-steps>
  local model="$1" etas="$2" msteps="$3"
  # NOTE: hellaswag/arc_easy moved to CPU-1 (pipeline_cpu1.sh) to parallelize.
  for task in sst2 boolq; do
    ml=128; [ "$task" = "boolq" ] && ml=384; [ "$task" = "hellaswag" -o "$task" = "arc_easy" ] && ml=256
    $PY scripts/phase1.py --model "$model" --task "$task" \
        --methods frozen lora mezo loca_f --seeds 0 \
        --n-train 2000 --n-eval 500 --max-len $ml --batch-size 8 \
        --device cpu --threads $TH --loca-etas $etas --T 12 --mezo-steps $msteps \
        --csv "$QCSV" || return 1
  done
}
stage q_05b threeway "Qwen/Qwen2.5-0.5B" "0.003 0.006" 3000
stage q_15b threeway "Qwen/Qwen2.5-1.5B" "0.01 0.02"   3000

# ---- (B) Large-model fp32 feasibility (3B/7B; OOM'd at 32GB, should fit 64GB) #
bigfeas(){
  for model in Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B; do
    for method in frozen lora loca_f mezo; do
      $PY scripts/phase2_efficiency.py --model "$model" --method "$method" --dtype float32 \
          --n-train 32 --max-len 128 --batch-size 8 --threads $TH --csv "$ECSV" || true
    done
  done
}
stage big_feasibility bigfeas

log "PIPELINE_CPU2_ALL_DONE"
