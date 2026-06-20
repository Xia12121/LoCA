#!/usr/bin/env bash
# Unattended GPU experiment chain (single-eta, fast): 0.5B -> 1.5B -> MeZO (C2).
# Single eta + T=8 (early-stop picks the best outer iter anyway). ~3x faster than
# the 2-eta/T=12 setup with negligible quality loss.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
PY=${PY:-/venv/main/bin/python}
ST=outputs/run_all_status.log
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$ST"; }

log "run_all(single-eta): starting 0.5B matrix"
bash scripts/phase1_matrix.sh "Qwen/Qwen2.5-0.5B" "0.003" 8 \
     "sst2 boolq alpaca" "frozen lora loca_f" "0 1 2" > outputs/phase1_05B.log 2>&1
log "run_all: 0.5B DONE -> starting 1.5B matrix"

bash scripts/phase1_matrix.sh "Qwen/Qwen2.5-1.5B" "0.002" 8 \
     "sst2 alpaca" "frozen lora loca_f" "0 1 2" > outputs/phase1_15B.log 2>&1
log "run_all: 1.5B DONE -> starting MeZO (C2, 0.5B sst2+alpaca seed0)"

for task in sst2 alpaca; do
  ml=128; [ "$task" = "alpaca" ] && ml=512
  log "run_all: MeZO $task"
  $PY scripts/phase1.py --model Qwen/Qwen2.5-0.5B --task "$task" \
      --methods mezo --seeds 0 --n-train 3000 --n-eval 800 --max-len $ml \
      --batch-size 16 --device cuda --mezo-steps 6000 >> outputs/mezo.log 2>&1
done
log "run_all: ALL STAGES DONE"
