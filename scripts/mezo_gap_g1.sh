#!/usr/bin/env bash
cd /workspace/loca; export HF_HOME=/workspace/.hf_home; PY=/venv/main/bin/python
ST=outputs/mezo_gap_g1_status.log; LOGS=outputs/logs; mkdir -p "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*"|tee -a "$ST"; }
log "wait FILL_15B_ALL_DONE..."; for i in $(seq 1 360); do grep -q FILL_15B_ALL_DONE outputs/fill_15b_status.log 2>/dev/null && break; sleep 60; done; log "go"
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local n="$1"; shift; done_stage "$n" && { log "$n SKIP"; return; }; log "$n START"; if "$@" >>"$LOGS/mgap_${n}.log" 2>&1; then log "STAGE_${n}_DONE"; else log "$n FAILED"; fi; }
c2(){ local m="$1" e="$2" T="$3" lr="$4" ml="$5" tsk="$6"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/c2_convergence.py --model "$m" --task "$tsk" \
    --device cuda --dtype bfloat16 --mezo-full-param --loca-etas $e --loca-T $T \
    --mezo-total-steps 10000 --mezo-eval-every 500 --mezo-lr $lr \
    --n-train 2000 --n-eval 500 --max-len $ml --batch-size 8 --csv outputs/c2_curve_scaling.csv; }
stage g1_15b_a c2 Qwen/Qwen2.5-1.5B "0.01 0.02" 40 5e-8 128 sst2
stage g1_15b_b c2 Qwen/Qwen2.5-1.5B "0.01 0.02" 40 3e-8 128 sst2
stage g1_3b_a  c2 Qwen/Qwen2.5-3B   "0.01 0.02" 30 3e-8 128 sst2
stage g1_3b_b  c2 Qwen/Qwen2.5-3B   "0.01 0.02" 30 1e-8 128 sst2
log "MEZO_GAP_G1_DONE"
