#!/usr/bin/env bash
# Fixes / supplements pipeline (GPU): re-tune 1.5B eta, fix Alpaca, C2 convergence
# (pull MeZO apart), Pythia scaling. Resumable + per-stage logs + snapshots.
# Writes to outputs/results_v2.csv (quality) and outputs/c2_curve.csv (C2 curves)
# so it never collides with the existing results.csv.
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
PY=${PY:-/venv/main/bin/python}
CSV=outputs/results_v2.csv
ST=outputs/pipeline_fixes_status.log
SNAP=outputs/snapshots; LOGS=outputs/logs
mkdir -p "$SNAP" "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name: SKIP"; return; fi
  local slog="$LOGS/fix_${name}.log"
  log "stage $name: START (detail -> $slog)"
  echo "===== $(date +%F_%T) $name =====" >> "$slog"
  if "$@" >> "$slog" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name: FAILED (continuing)"; fi
  cp -f "$CSV" "$SNAP/results_v2_after_$name.csv" 2>/dev/null
}

# ---- FIX 1a: 1.5B de-risk PROBE (seed0) — larger eta + larger T ----------- #
# T=12 was not converged (best_t=12) at small eta. Probe larger etas (faster
# descent) with T=40 + early stop. If recovery climbs, expand to 3 seeds (1b).
stage fix_15b_probe $PY scripts/phase1.py --model Qwen/Qwen2.5-1.5B --task sst2 \
      --methods frozen lora loca_f --seeds 0 --n-train 3000 --n-eval 800 --max-len 128 \
      --batch-size 16 --device cuda --loca-etas 0.01 0.02 --T 40 --csv "$CSV"
# 1b: full 3-seed 1.5B (runs only after probe confirms scaling; cheap to re-tag)
stage fix_15b_full $PY scripts/phase1.py --model Qwen/Qwen2.5-1.5B --task sst2 \
      --methods frozen lora loca_f --seeds 1 2 --n-train 3000 --n-eval 800 --max-len 128 \
      --batch-size 16 --device cuda --loca-etas 0.01 0.02 --T 40 --csv "$CSV"

# ---- FIX 2: Alpaca (lower LoRA lr + fewer epochs) ------------------------- #
stage fix_alpaca $PY scripts/phase1.py --model Qwen/Qwen2.5-0.5B --task alpaca \
      --methods frozen lora loca_f --seeds 0 1 2 --n-train 4000 --n-eval 500 --max-len 512 \
      --batch-size 8 --device cuda --loca-etas 0.002 0.005 --T 10 \
      --lora-lr 5e-5 --lora-epochs 2 --csv "$CSV"

# ---- FIX 3: C2 convergence (LoCA vs MeZO, quality-vs-walltime) ------------ #
stage c2_05b_sst2 $PY scripts/c2_convergence.py --model Qwen/Qwen2.5-0.5B --task sst2 \
      --device cuda --loca-etas 0.003 --loca-T 8 --mezo-total-steps 12000 --mezo-eval-every 500
stage c2_15b_sst2 $PY scripts/c2_convergence.py --model Qwen/Qwen2.5-1.5B --task sst2 \
      --device cuda --loca-etas 0.001 0.002 --loca-T 12 --mezo-total-steps 12000 --mezo-eval-every 500

# ---- NEW BENCHMARKS: commonsense suite (PEFT standard) on 0.5B ------------ #
newbench(){
  for task in hellaswag arc_easy arc_challenge; do
    $PY scripts/phase1.py --model Qwen/Qwen2.5-0.5B --task "$task" \
        --methods frozen lora loca_f --seeds 0 1 2 --n-train 3000 --n-eval 800 \
        --max-len 256 --batch-size 8 --device cuda --loca-etas 0.003 0.006 --T 10 \
        --csv "$CSV" || return 1
  done
}
stage newbench_05b newbench

# ---- FIX 4: Pythia scaling (recovery vs model size, clean family) --------- #
pythia_scaling(){
  for M in EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b; do
    $PY scripts/phase1.py --model "$M" --task sst2 --methods frozen lora loca_f --seeds 0 \
        --n-train 2000 --n-eval 600 --max-len 128 --batch-size 16 --device cuda \
        --loca-etas 0.002 0.005 0.01 --T 10 --csv "$CSV" || return 1
  done
}
stage pythia_scaling pythia_scaling

log "PIPELINE_FIXES_ALL_DONE"
