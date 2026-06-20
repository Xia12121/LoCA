#!/usr/bin/env bash
# GPU-3 pipeline (parallel accelerator) — takes the slowest / most-valuable items
# off GPU-1 & GPU-2's critical path:
#   (1) C2 convergence (MeZO vs LoCA, quality-vs-walltime) at 0.5B / 1.5B / 7B
#       -> directly tests the "MeZO variance ∝ D" theory across model size.
#   (2) Pythia scaling curve (recovery vs size).
# Own CSVs (c2_curve_g3.csv / results_pythia_g3.csv) — never collides.
#
# Launch: cd /workspace/loca && HF_HOME=/workspace/.hf_home \
#         setsid bash scripts/pipeline_gpu3.sh > outputs/pipeline_gpu3.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
PY=${PY:-/venv/main/bin/python}
ST=outputs/pipeline_gpu3_status.log
LOGS=outputs/logs; mkdir -p "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name: SKIP"; return; fi
  local slog="$LOGS/g3_${name}.log"
  log "stage $name: START (detail -> $slog)"
  echo "===== $(date +%F_%T) $name =====" >> "$slog"
  if "$@" >> "$slog" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name: FAILED (continuing)"; fi
}
C2=outputs/c2_curve_g3.csv

# ---- (1) C2 convergence at three sizes (the MeZO foil) -------------------- #
stage c2_05b $PY scripts/c2_convergence.py --model Qwen/Qwen2.5-0.5B --task sst2 \
      --device cuda --loca-etas 0.003 0.006 --loca-T 10 --mezo-total-steps 12000 \
      --mezo-eval-every 500 --csv "$C2"
stage c2_15b $PY scripts/c2_convergence.py --model Qwen/Qwen2.5-1.5B --task sst2 \
      --device cuda --loca-etas 0.01 0.02 --loca-T 15 --mezo-total-steps 12000 \
      --mezo-eval-every 500 --csv "$C2"
stage c2_7b  $PY scripts/c2_convergence.py --model Qwen/Qwen2.5-7B --task sst2 \
      --device cuda --loca-etas 0.01 0.02 --loca-T 15 --mezo-total-steps 10000 \
      --mezo-eval-every 500 --csv "$C2"

# ---- (2) Pythia scaling curve (recovery vs size) ------------------------- #
pythia(){
  for M in EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b \
           EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b; do
    $PY scripts/phase1.py --model "$M" --task sst2 --methods frozen lora loca_f --seeds 0 \
        --n-train 2000 --n-eval 600 --max-len 128 --batch-size 16 --device cuda \
        --loca-etas 0.005 0.01 0.02 --T 20 --csv outputs/results_pythia_g3.csv || return 1
  done
}
stage pythia_scaling pythia

log "PIPELINE_GPU3_ALL_DONE"
