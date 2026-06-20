#!/usr/bin/env bash
# GPU-2 pipeline (NEW instance) — scaling headline: 7B quality + Pythia scaling.
# Writes to results_7b.csv / results_pythia.csv / c2_curve_g2.csv (own files, never
# collides with GPU-1). Resumable + per-stage logs + snapshots.
#
# Launch: cd /workspace/loca && HF_HOME=/workspace/.hf_home \
#         setsid bash scripts/pipeline_gpu2.sh > outputs/pipeline_gpu2.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
PY=${PY:-/venv/main/bin/python}
ST=outputs/pipeline_gpu2_status.log
SNAP=outputs/snapshots; LOGS=outputs/logs
mkdir -p "$SNAP" "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name: SKIP"; return; fi
  local slog="$LOGS/g2_${name}.log"
  log "stage $name: START (detail -> $slog)"
  echo "===== $(date +%F_%T) $name =====" >> "$slog"
  if "$@" >> "$slog" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name: FAILED (continuing)"; fi
}

# eta for 7B/1.5B-class (28 layers): narrowed to the validated sweet spot (speed).
ETAS_BIG="0.01 0.02"
ETAS_SMALL="0.005 0.01"   # smaller models

# ---- 7B quality (bf16) — the headline scaling proof ----------------------- #
stage q_7b_sst2 $PY scripts/phase1.py --model Qwen/Qwen2.5-7B --task sst2 \
      --methods frozen lora loca_f --seeds 0 --n-train 3000 --n-eval 800 --max-len 128 \
      --batch-size 8 --device cuda --dtype bfloat16 --loca-etas $ETAS_BIG --T 40 \
      --csv outputs/results_7b.csv
stage q_7b_boolq $PY scripts/phase1.py --model Qwen/Qwen2.5-7B --task boolq \
      --methods frozen lora loca_f --seeds 0 --n-train 2000 --n-eval 600 --max-len 512 \
      --batch-size 4 --device cuda --dtype bfloat16 --loca-etas $ETAS_BIG --T 40 \
      --csv outputs/results_7b.csv

# ---- Pythia scaling curve (clean same-family, recovery vs size) ----------- #
pythia(){
  for M in EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b \
           EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b; do
    $PY scripts/phase1.py --model "$M" --task sst2 --methods frozen lora loca_f --seeds 0 \
        --n-train 2000 --n-eval 600 --max-len 128 --batch-size 16 --device cuda \
        --loca-etas $ETAS_SMALL --T 20 --csv outputs/results_pythia.csv || return 1
  done
}
stage pythia_scaling pythia

# ---- 7B commonsense suite (PEFT standard benchmarks) ---------------------- #
newbench_7b(){
  for task in hellaswag arc_easy arc_challenge; do
    $PY scripts/phase1.py --model Qwen/Qwen2.5-7B --task "$task" \
        --methods frozen lora loca_f --seeds 0 --n-train 2000 --n-eval 600 \
        --max-len 256 --batch-size 4 --device cuda --dtype bfloat16 \
        --loca-etas 0.01 0.02 --T 30 --csv outputs/results_7b.csv || return 1
  done
}
stage newbench_7b newbench_7b

# ---- C2 convergence on 7B (where ZO variance hurts most) ------------------ #
stage c2_7b $PY scripts/c2_convergence.py --model Qwen/Qwen2.5-7B --task sst2 \
      --device cuda --loca-etas 0.01 0.02 --loca-T 20 --mezo-total-steps 8000 \
      --mezo-eval-every 500 --csv outputs/c2_curve_g2.csv

log "PIPELINE_GPU2_ALL_DONE"
