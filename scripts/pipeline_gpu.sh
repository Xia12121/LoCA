#!/usr/bin/env bash
# Master GPU pipeline — runs ALL remaining GPU experiments unattended, server-side.
# Resilient to disconnection: setsid + per-cell resume (phase1 skips done rows) +
# stage-level resume (skips stages already marked DONE) + a CSV snapshot after every
# stage. A failed stage is logged and the pipeline continues.
#
# Launch: cd /workspace/loca && HF_HOME=/workspace/.hf_home \
#         setsid bash scripts/pipeline_gpu.sh > outputs/pipeline_gpu.log 2>&1 < /dev/null &
# Re-launch the SAME command after a crash: completed stages/cells are skipped.
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
PY=${PY:-/venv/main/bin/python}
CSV=outputs/results.csv
ST=outputs/pipeline_status.log
SNAP=outputs/snapshots
LOGS=outputs/logs
mkdir -p "$SNAP" "$LOGS"
touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
snap(){ cp -f "$CSV" "$SNAP/results_after_$1.csv" 2>/dev/null && log "snapshot -> $SNAP/results_after_$1.csv"; }
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){  # stage <name> <command...>
  local name="$1"; shift
  if done_stage "$name"; then log "stage $name: SKIP (already done)"; return; fi
  local slog="$LOGS/${name}.log"
  log "stage $name: START (detail -> $slog)"
  echo "===== $(date +%F_%T) $name START =====" >> "$slog"
  if "$@" >> "$slog" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name: FAILED (continuing)"; fi
  snap "$name"
}

PM="bash scripts/phase1_matrix.sh"

# ---- Phase 1 (quality) ---------------------------------------------------- #
stage p1_05B  $PM "Qwen/Qwen2.5-0.5B" "0.003" 8 "sst2 boolq alpaca" "frozen lora loca_f" "0 1 2"
stage p1_15B  $PM "Qwen/Qwen2.5-1.5B" "0.002" 8 "sst2 alpaca"        "frozen lora loca_f" "0 1 2"

# ---- Phase 1 C2: MeZO competitor ------------------------------------------ #
mezo_stage(){
  for task in sst2 alpaca; do
    ml=128; [ "$task" = "alpaca" ] && ml=512
    $PY scripts/phase1.py --model Qwen/Qwen2.5-0.5B --task "$task" --methods mezo \
        --seeds 0 --n-train 3000 --n-eval 800 --max-len $ml --batch-size 16 \
        --device cuda --mezo-steps 6000 --csv "$CSV" || return 1
  done
}
stage p1_mezo mezo_stage

# ---- Phase 3 (science, C4) ------------------------------------------------- #
# Negative control: GSM8K should NOT recover (long-range credit assignment).
stage p3_gsm8k_negctrl $PY scripts/phase1.py --model Qwen/Qwen2.5-0.5B --task gsm8k \
      --methods frozen lora loca_f --seeds 0 --n-train 2000 --n-eval 400 --max-len 512 \
      --batch-size 8 --device cuda --loca-etas 0.003 --T 8 --csv "$CSV"
# Alignment-angle curve (random vs sketch) on Pythia (clean scaling family).
stage p3_align  $PY scripts/phase3_science.py --exp align  --override model.name=EleutherAI/pythia-410m train.dataset=sst2 train.n_train=1000 train.max_len=128
# eta failure boundary.
stage p3_eta    $PY scripts/phase3_science.py --exp eta    --override model.name=Qwen/Qwen2.5-0.5B train.dataset=sst2 train.n_train=1500 train.max_len=128
# linearization O(eta^2) gap.
stage p3_lingap $PY scripts/phase3_science.py --exp lingap --override model.name=Qwen/Qwen2.5-0.5B train.dataset=sst2 train.n_train=512 train.max_len=128

# ---- Phase 4 (ablations, C5) ---------------------------------------------- #
# Jacobi vs Gauss-Seidel (GS run tagged as task sst2_gs).
stage p4_gs $PY scripts/phase1.py --model Qwen/Qwen2.5-0.5B --task sst2 --methods loca_f \
      --seeds 0 1 2 --n-train 3000 --n-eval 800 --max-len 128 --batch-size 16 --device cuda \
      --loca-etas 0.003 --T 8 --mode gauss_seidel --task-name sst2_gs --csv "$CSV"
# r / lam / T sensitivity sweeps.
sweep_stage(){
  $PY scripts/sweep.py --config configs/loca_f.yaml --param adapter.r --values 16 32 64 \
      --override model.name=Qwen/Qwen2.5-0.5B model.device=cuda train.dataset=sst2 train.n_train=1500 train.max_len=128 loca.feedback=sketch loca.eta=0.003 loca.T=8 'runtime.seeds=[0]' --task-name sweep_r || return 1
  $PY scripts/sweep.py --config configs/loca_f.yaml --param loca.lam --values 0.01 0.1 1.0 \
      --override model.name=Qwen/Qwen2.5-0.5B model.device=cuda train.dataset=sst2 train.n_train=1500 train.max_len=128 loca.feedback=sketch loca.eta=0.003 loca.T=8 'runtime.seeds=[0]' --task-name sweep_lam || return 1
  $PY scripts/sweep.py --config configs/loca_f.yaml --param loca.T --values 1 3 5 10 \
      --override model.name=Qwen/Qwen2.5-0.5B model.device=cuda train.dataset=sst2 train.n_train=1500 train.max_len=128 loca.feedback=sketch loca.eta=0.003 'runtime.seeds=[0]' --task-name sweep_T || return 1
}
stage p4_sweep sweep_stage

log "PIPELINE_GPU_ALL_DONE"
