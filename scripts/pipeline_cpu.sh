#!/usr/bin/env bash
# Master CPU pipeline — proves LoCA trains to good QUALITY on the no-GPU box
# (the headline: "post-training where backprop is infeasible"), plus re-runs the
# efficiency sweep if missing. Resumable + per-stage CSV snapshots, server-side.
#
# Launch: cd /data/loca && HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data/hf_cache \
#         setsid bash scripts/pipeline_cpu.sh > outputs/pipeline_cpu.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/hf_cache}
export OMP_NUM_THREADS=16
PY=.venv/bin/python
QCSV=outputs/cpu_quality.csv
ECSV=outputs/efficiency.csv
ST=outputs/pipeline_cpu_status.log
SNAP=outputs/snapshots
LOGS=outputs/logs
mkdir -p "$SNAP" "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
snap(){ cp -f "$1" "$SNAP/$(basename $1 .csv)_after_$2.csv" 2>/dev/null && log "snapshot $1 -> $2"; }
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name: SKIP"; return; fi
  local slog="$LOGS/${name}.log"
  log "stage $name: START (detail -> $slog)"
  echo "===== $(date +%F_%T) $name START =====" >> "$slog"
  if "$@" >> "$slog" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name: FAILED (continuing)"; fi
}

# ---- CPU quality (real task, threads=16) — proves quality on a no-GPU box --- #
cpu_quality(){  # cpu_quality <model> <eta> <n_train>
  $PY scripts/phase1.py --model "$1" --task sst2 --methods frozen lora loca_f \
      --seeds 0 --n-train "$3" --n-eval 600 --max-len 128 --batch-size 16 \
      --device cpu --threads 16 --loca-etas "$2" --T 8 --csv "$QCSV"
}
stage cpu_q_05B "cpu_quality" "Qwen/Qwen2.5-0.5B" 0.003 2000; snap "$QCSV" cpu_q_05B
stage cpu_q_15B "cpu_quality" "Qwen/Qwen2.5-1.5B" 0.002 1500; snap "$QCSV" cpu_q_15B

# ---- Efficiency sweep (idempotent; skips if efficiency.csv already populated) #
eff_sweep(){
  [ -s "$ECSV" ] && [ "$(wc -l < $ECSV)" -gt 20 ] && return 0   # already done
  bash scripts/phase2_matrix.sh "Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B" \
       "frozen lora loca_f mezo" 32 128 16
}
stage cpu_eff eff_sweep; snap "$ECSV" cpu_eff

log "PIPELINE_CPU_ALL_DONE"
