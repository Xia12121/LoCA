#!/usr/bin/env bash
# GPU-1 third stage: 14B (~"20B class") full benchmark + C2 (full-MeZO) to widen LoCA-vs-MeZO gap.
cd /workspace/loca; export HF_HOME=/workspace/.hf_home; PY=/venv/main/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ST=outputs/gpu1_14b_status.log; LOGS=outputs/logs; mkdir -p "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*"|tee -a "$ST"; }
log "wait MEZO_GAP_G1_DONE..."; for i in $(seq 1 900); do grep -q MEZO_GAP_G1_DONE outputs/mezo_gap_g1_status.log 2>/dev/null && break; sleep 60; done; log "go 14B"
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local n="$1"; shift; done_stage "$n" && { log "$n SKIP"; return; }; log "$n START"; if "$@" >>"$LOGS/g14_${n}.log" 2>&1; then log "STAGE_${n}_DONE"; else log "$n FAILED(continuing)"; fi; }
# ---- FULL benchmark: 5 tasks x frozen/lora/mezo/loca (single eta 0.01 to keep 14B tractable) ----
bench(){ local tsk="$1" ml="$2"
  $PY scripts/phase1.py --model Qwen/Qwen2.5-14B --task "$tsk" --methods frozen lora mezo loca_f --seeds 0 \
    --n-train 2000 --n-eval 500 --max-len $ml --batch-size 2 --device cuda --dtype bfloat16 \
    --loca-etas 0.01 --T 30 --mezo-steps 3000 --csv outputs/results_14b.csv; }
stage b14_sst2          bench sst2          128
stage b14_boolq         bench boolq         384
stage b14_arc_easy      bench arc_easy      256
stage b14_arc_challenge bench arc_challenge 256
stage b14_hellaswag     bench hellaswag     256
# ---- C2: LoCA vs FULL-param MeZO (the gap headline; D=14B) ----
c2(){ local tsk="$1" ml="$2" lr="$3"
  $PY scripts/c2_convergence.py --model Qwen/Qwen2.5-14B --task "$tsk" --device cuda --dtype bfloat16 \
    --mezo-full-param --loca-etas 0.01 0.02 --loca-T 15 --mezo-total-steps 10000 --mezo-eval-every 1000 --mezo-lr $lr \
    --n-train 2000 --n-eval 500 --max-len $ml --batch-size 4 --csv outputs/c2_curve_14b.csv; }
stage c14_sst2_a c2 sst2     128 1e-8
stage c14_sst2_b c2 sst2     128 5e-9
stage c14_boolq  c2 boolq    384 1e-8
stage c14_arce   c2 arc_easy 256 1e-8
log "GPU1_14B_DONE"
