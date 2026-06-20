#!/usr/bin/env bash
cd /workspace/loca
export HF_HOME=/workspace/.hf_home
PY=/venv/main/bin/python
ST=outputs/gpu1_fill_status.log
LOGS=outputs/logs; mkdir -p "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name SKIP"; return; fi
  log "stage $name START (-> $LOGS/g1fill_${name}.log)"
  if "$@" >> "$LOGS/g1fill_${name}.log" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name FAILED (continuing)"; fi
}
# A: GPU memory+time efficiency (peak + STEADY + base VRAM, per-pass wall) all sizes x methods
ECSV=outputs/efficiency_gpu.csv
eff(){ for model in Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B; do
  for method in frozen lora mezo loca_f; do
    log "  eff $model $method"
    $PY scripts/phase2_efficiency.py --model "$model" --method "$method" --device cuda \
        --dtype bfloat16 --n-train 512 --max-len 256 --batch-size 8 --csv "$ECSV" || true
  done; done; }
stage gpu_efficiency eff
# B: 7B MeZO quality benchmarks (pairs with results_7b frozen/lora, matched config)
QCSV=outputs/results_7b_mezo.csv
mezo7b(){ for task in boolq arc_easy arc_challenge hellaswag; do
  ml=256; [ "$task" = "boolq" ] && ml=384
  $PY scripts/phase1.py --model Qwen/Qwen2.5-7B --task "$task" \
    --methods frozen mezo --seeds 0 \
    --n-train 2000 --n-eval 600 --max-len $ml --batch-size 4 \
    --device cuda --dtype bfloat16 --mezo-steps 3000 --csv "$QCSV" || return 1; done; }
stage mezo_7b_bench mezo7b
log "GPU1_FILL_ALL_DONE"
