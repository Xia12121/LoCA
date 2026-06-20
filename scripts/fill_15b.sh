#!/usr/bin/env bash
# 1.5B coverage fill: the tasks 0.5B has but 1.5B doesn't yet
# (hellaswag, winogrande, obqa + negative controls alpaca/gsm8k). All 4 methods.
# Waits for GPU-1's efficiency+7Bmezo job to finish, then runs. Resumable.
cd /workspace/loca
export HF_HOME=/workspace/.hf_home
PY=/venv/main/bin/python
CSV=outputs/results_15b_fill.csv
ST=outputs/fill_15b_status.log
WAIT_ST=outputs/gpu1_fill_status.log
LOGS=outputs/logs; mkdir -p "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
log "waiting for GPU1_FILL_ALL_DONE ..."
for i in $(seq 1 240); do
  grep -q "GPU1_FILL_ALL_DONE" "$WAIT_ST" 2>/dev/null && { log "prev job done -> start 1.5B fill"; break; }
  sleep 60
done
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name SKIP"; return; fi
  log "stage $name START (-> $LOGS/fill15b_${name}.log)"
  if "$@" >> "$LOGS/fill15b_${name}.log" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name FAILED (continuing)"; fi
}
run(){ local task="$1" ml="$2"
  $PY scripts/phase1.py --model Qwen/Qwen2.5-1.5B --task "$task" \
    --methods frozen lora mezo loca_f --seeds 0 \
    --n-train 2000 --n-eval 500 --max-len $ml --batch-size 8 \
    --device cuda --dtype float32 --loca-etas 0.01 0.02 --T 40 --mezo-steps 3000 --csv "$CSV"; }
stage f_hellaswag     run hellaswag    256
stage f_winogrande    run winogrande   256
stage f_obqa          run obqa         256
stage f_arc_challenge run arc_challenge 256
log "FILL_15B_ALL_DONE"
