# Reproduction guide

All runs write one row per (seed, metric) to `outputs/results.csv`. Figures are
regenerated from that CSV. Fix seeds via `runtime.seeds` (default `[0,1,2]`).

## 0. Environment
```bash
./setup_env.sh           # CPU dev box or GPU box (auto-detects torch wheel)
source .venv/bin/activate
```

## Phase 0 — mechanism gate (CPU, minutes)  [must pass before anything else]
```bash
python scripts/phase0.py --model gpt2 --feedback sketch      # E0.1/E0.2/E0.3
python -m pytest tests/ -q                                   # unit + mechanism
```

## Phase 1 — quality (C1/C2)
Run every method on a task, then recovery comes from the CSV (frozen & lora rows
must exist). Example on Qwen2.5-0.5B / instruction SFT:
```bash
M=Qwen/Qwen2.5-0.5B; T=alpaca
python scripts/run_baseline.py --config configs/baselines/full_sft.yaml --override method=frozen model.name=$M --task-name $T
python scripts/run_baseline.py --config configs/baselines/lora.yaml      --override model.name=$M --task-name $T
python scripts/run_baseline.py --config configs/baselines/mezo.yaml      --override model.name=$M --task-name $T
python scripts/run_loca.py     --config configs/loca_d.yaml              --override model.name=$M loca.feedback=sketch --task-name $T
python scripts/run_loca.py     --config configs/loca_f.yaml              --override model.name=$M loca.feedback=sketch --task-name $T
```
Gate: LoCA-F recovery ≥ 0.85 and > MeZO. Clean scalar tasks first
(`train.dataset=sst2` or `format_json`), then `alpaca`.

## Phase 2 — efficiency & CPU headline (C3)
Same machine, same dtype/threads for all methods. Sweep model size:
```bash
for M in Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B-Instruct; do
  python scripts/run_loca.py     --config configs/loca_f.yaml         --override model.name=$M model.device=cpu loca.feedback=sketch --task-name alpaca
  python scripts/run_baseline.py --config configs/baselines/lora.yaml --override model.name=$M model.device=cpu --task-name alpaca
  python scripts/run_baseline.py --config configs/baselines/mezo.yaml --override model.name=$M model.device=cpu --task-name alpaca
done
python scripts/make_figures.py        # cpu_scaling.png
```
Gate: 7–8B CPU LoCA wall-clock < LoRA/5 and RAM < 1/3.

## Phase 3 — science (C4)
```bash
python scripts/phase3_science.py --exp align  --override model.name=EleutherAI/pythia-410m
python scripts/phase3_science.py --exp eta     --override model.name=EleutherAI/pythia-410m
python scripts/phase3_science.py --exp lingap  --override model.name=EleutherAI/pythia-410m
# negative control: GSM8K should NOT recover
python scripts/run_loca.py --config configs/loca_f.yaml --override train.dataset=gsm8k --task-name gsm8k_negctrl
```

## Phase 4 — ablations (C5)
```bash
python scripts/sweep.py --config configs/loca_f.yaml --param loca.eta --values 0.01 0.02 0.05 0.1
python scripts/sweep.py --config configs/loca_f.yaml --param loca.lam --values 0.01 0.1 1.0
python scripts/sweep.py --config configs/loca_f.yaml --param loca.T   --values 1 3 5 10
python scripts/sweep.py --config configs/loca_f.yaml --param adapter.r --values 16 32 64
python scripts/run_loca.py --config configs/loca_f.yaml --override loca.mode=gauss_seidel   # vs jacobi
python scripts/run_loca.py --config configs/loca_f.yaml --override loca.feedback=random     # vs sketch
```

## Figures
```bash
python scripts/make_figures.py --csv outputs/results.csv --out outputs/figures
```

## Notes
- `feedback=sketch` is the working variant; `random` is the ablation foil.
- Efficiency numbers must be collected in one session on one machine (§6 fairness).
- MMLU / IFEval via `src/eval/run_lm_eval.py` (needs `lm-eval` on the GPU box).
