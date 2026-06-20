#!/usr/bin/env bash
# GPU-1 MeZO SUPPLEMENT pipeline: fill in MeZO baselines on benchmarks that the
# main runs only had frozen/lora/loca for. Each task runs frozen+lora+mezo in the
# SAME csv so recovery is internally consistent (matched n_train/max_len).
# Own CSV results_mezo_g1.csv. Resumable: phase1 skips done (method,model,task,seed);
# stage markers skip whole stages.
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
PY=${PY:-/venv/main/bin/python}
CSV=outputs/results_mezo_g1.csv
ST=outputs/pipeline_gpu1_mezo_status.log
SNAP=outputs/snapshots; LOGS=outputs/logs
mkdir -p "$SNAP" "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name: SKIP"; return; fi
  local slog="$LOGS/g1mezo_${name}.log"
  log "stage $name: START (detail -> $slog)"
  echo "===== $(date +%F_%T) $name =====" >> "$slog"
  if "$@" >> "$slog" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name: FAILED (continuing)"; fi
  cp -f "$CSV" "$SNAP/results_mezo_g1_after_$name.csv" 2>/dev/null
}
# run frozen+lora+mezo on one model/task with matched config
trio(){ local model="$1" task="$2" ml="$3" seeds="$4" etas="$5" T="$6"
  $PY scripts/phase1.py --model "$model" --task "$task" \
      --methods frozen lora mezo loca_f --seeds $seeds \
      --n-train 2000 --n-eval 500 --max-len $ml --batch-size 8 \
      --device cuda --loca-etas $etas --T $T --mezo-steps 10000 --csv "$CSV"
}
# ---- 0.5B benchmarks missing MeZO (3 seeds) ----
stage mezo_05b_boolq         trio Qwen/Qwen2.5-0.5B boolq         384 "0 1 2" "0.003 0.006" 12
stage mezo_05b_arc_easy      trio Qwen/Qwen2.5-0.5B arc_easy      256 "0 1 2" "0.003 0.006" 12
stage mezo_05b_arc_challenge trio Qwen/Qwen2.5-0.5B arc_challenge 256 "0 1 2" "0.003 0.006" 12
stage mezo_05b_hellaswag     trio Qwen/Qwen2.5-0.5B hellaswag     256 "0 1 2" "0.003 0.006" 12
# ---- 1.5B (1 seed first; slower) ----
stage mezo_15b_sst2          trio Qwen/Qwen2.5-1.5B sst2          128 "0" "0.01 0.02" 40
stage mezo_15b_boolq         trio Qwen/Qwen2.5-1.5B boolq         384 "0" "0.01 0.02" 40
log "PIPELINE_GPU1_MEZO_ALL_DONE"
