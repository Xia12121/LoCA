#!/usr/bin/env bash
# Accuracy pipeline (GPU-1): wires the new rank-classification accuracy into the
# discriminative suite. Waits for the MeZO supplement to finish, then runs
# frozen/lora/mezo/loca on sst2/boolq/arc_easy/arc_challenge across sizes.
# Fresh CSV results_acc.csv -> every cell recomputed WITH metric=acc (+ce). Resumable.
cd /workspace/loca
export HF_HOME=/workspace/.hf_home
PY=/venv/main/bin/python
CSV=outputs/results_acc.csv
ST=outputs/pipeline_acc_status.log
MEZO_ST=outputs/pipeline_gpu1_mezo_status.log
LOGS=outputs/logs; mkdir -p "$LOGS"; touch "$ST"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }
log "waiting for MeZO supplement (PIPELINE_GPU1_MEZO_ALL_DONE) ..."
for i in $(seq 1 480); do
  grep -q "PIPELINE_GPU1_MEZO_ALL_DONE" "$MEZO_ST" 2>/dev/null && { log "MeZO supplement done -> starting accuracy"; break; }
  sleep 60
done
done_stage(){ grep -q "STAGE_$1_DONE" "$ST"; }
stage(){ local name="$1"; shift
  if done_stage "$name"; then log "stage $name SKIP"; return; fi
  log "stage $name START (-> $LOGS/acc_${name}.log)"
  if "$@" >> "$LOGS/acc_${name}.log" 2>&1; then log "STAGE_${name}_DONE"; else log "stage $name FAILED (continuing)"; fi
}
acc(){  # acc <model> <etas> <T> <dtype> <bs>
  local model="$1" etas="$2" T="$3" dt="$4" bs="$5"
  for task in sst2 boolq arc_easy arc_challenge; do
    ml=256; [ "$task" = "sst2" ] && ml=128; [ "$task" = "boolq" ] && ml=384
    $PY scripts/phase1.py --model "$model" --task "$task" \
      --methods frozen lora mezo loca_f --seeds 0 \
      --n-train 2000 --n-eval 500 --max-len $ml --batch-size $bs \
      --device cuda --dtype "$dt" --loca-etas $etas --T $T --mezo-steps 3000 \
      --csv "$CSV" || return 1
  done
}
stage acc_05b acc Qwen/Qwen2.5-0.5B "0.003 0.006" 12 float32 8
stage acc_15b acc Qwen/Qwen2.5-1.5B "0.01 0.02"   40 float32 8
stage acc_7b  acc Qwen/Qwen2.5-7B   "0.01 0.02"   40 bfloat16 4
log "PIPELINE_ACC_ALL_DONE"
