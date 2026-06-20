# LoCA — Local Credit Assignment for Post-Training

Backprop-free, block-parallel, CPU-feasible LLM post-training. Each transformer
block's low-rank adapter is solved in **closed form** (block-wise ridge regression)
against a **local target** formed from the top-layer error projected through a
**fixed feedback operator** (DFA-style). No global backward chain, no cross-layer
activation graph.

See `../LoCA_局部信用分配后训练_方法设计.md` (math, §4–6) and
`../LoCA_实验设计_AAAI27.md` (experiment spec) for the authoritative design.

## Layout
```
configs/        base + per-method YAML (loca_f, loca_d, baselines/*)
src/adapters/   ResidualLoRA (additive residual-stream low-rank) + model introspection
src/loca/       hooks, top_error, feedback, closed_form, solver, diagnostics, runner
src/baselines/  lora_sft, mezo, full_sft, frozen
src/eval/       perplexity, format_acc, recovery, run_lm_eval
scripts/        phase0, run_loca, run_baseline, sweep, phase3_science, make_figures
tests/          pytest unit + Phase-0 mechanism checks
```

## Core method (one screen)
- `h_l = f_l^base(h_{l-1}) + B_l A_l s_l`, with `s_l = h_{l-1}`, `A` frozen random,
  `B` solved in closed form. Additive on the **block output residual stream** so the
  closed form is exact (`src/adapters/residual_lora.py`).
- Top error `e = W_unembed^T(softmax(z) - onehot(y))`, LM head only, no backbone
  backward (`src/loca/top_error.py`).
- Local target `tau_l = h_l - eta F_l e`; residual `rho_l = B_l^{(t)} A_l s - eta F_l e`.
- Streaming Gram `G_l += p p^T`, `C_l += rho p^T`; solve `B_l = C_l (G_l + lam I)^{-1}`
  via `torch.linalg.solve` in float64 (`src/loca/closed_form.py`).
- Outer block-coordinate iteration, Jacobi (parallel) or Gauss-Seidel
  (`src/loca/solver.py`).

## Quick start
```bash
./setup_env.sh                 # or: source .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests/ -q     # 20 unit + mechanism tests
python scripts/phase0.py --model gpt2 --feedback sketch   # GO/NO-GO mechanism gate
```

## Status
- Phase 0 (mechanism) **passes** on gpt2 with `feedback=sketch`: alignment
  `cos(F e, g) > 0`, global CE decreases monotonically, held-out CE beats frozen.
- `feedback=random` (vanilla DFA) shows ~0 alignment on a frozen LLM backbone — the
  sketch variant is the working method (this is the Risk-A / C4 story).

Phases 1–4 (quality / CPU headline / science curves / ablations) run on the GPU box
via the scripts above; see `REPRODUCE.md`.
