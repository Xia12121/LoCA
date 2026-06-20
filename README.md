<div align="center">

# LoCA

### Local Credit Assignment for Backpropagation-Free LLM Post-Training

*Each transformer block learns by **solving a linear system**, not by descending a gradient.*

<br>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Backprop](https://img.shields.io/badge/backprop-free-2ea44f)
![Status](https://img.shields.io/badge/status-research-blue)

</div>

---

## What is LoCA?

LoCA post-trains a frozen LLM **without ever running the global backward pass.**
Instead of backpropagating one loss through all $L$ transformer blocks, it gives
each block its own **local target** and fits a small adapter to that target in
**closed form** (block-wise ridge regression).

The result is a training loop that is:

- **Backprop-free** — only forward passes + a top-layer error signal; no
  cross-layer activation graph, no optimizer state.
- **Block-parallel** — every block's adapter is an independent least-squares
  solve (embarrassingly parallel).
- **Deterministic** — closed-form, no learning-rate schedule, no random gradient
  estimate; the same data gives the same model.
- **Memory-light & CPU-feasible** — training memory is independent of the number
  of tokens (streaming Gram matrices), so it runs where backprop OOMs.

---

## The idea in one picture

```
                  ┌─────────────────── frozen backbone ───────────────────┐
   x ──► [blk 1]──►[blk 2]──► … ──►[blk L]──► LM head ──► logits z ──► CE loss
            │         │              │                                  │
            │         │              │                          e = ∂CE/∂h_L
            ▼         ▼              ▼            (LM head only, NO backbone backward)
          F₁·e      F₂·e          F_L·e   ◄───────────────────────────┘
            │         │              │      fixed feedback operator F_ℓ
            ▼         ▼              ▼      broadcasts the SAME top error e
        τ₁=h₁−ηF₁e  …            τ_L=h_L−ηF_L e          to every block
            │         │              │
            ▼         ▼              ▼
        solve B₁    solve B₂      solve B_L      ◄── closed-form ridge, in parallel
```

There is **no top-to-bottom arrow running back through the backbone** — that
chained Jacobian product is exactly what LoCA removes.

---

## How it works

### 1. The adapter (placed so the solve is *exact*)

Each block gets an additive low-rank correction on its **output residual stream**:

$$h_\ell = f_\ell^{\text{base}}(h_{\ell-1}) + B_\ell A_\ell\, s_\ell, \qquad s_\ell = h_{\ell-1}$$

| symbol | shape | role |
|---|---|---|
| $A_\ell$ | $r \times d$ | **frozen** random projection ($A\sim\mathcal N(0,1/d)$) |
| $B_\ell$ | $d \times r$ | **trainable**, solved in closed form; initialised to $0$ |
| $r \ll d$ | — | rank (we use 16 / 32 / 64) |

Because the correction is added **directly** to $h_\ell$ (it does not pass back
through the block's attention/MLP nonlinearity), $h_\ell$ is **exactly linear in
$B_\ell$** — which is what turns the per-block fit into an *exact* least-squares
problem. With $B_\ell=0$ the block equals the frozen block.

### 2. Why backprop is the bottleneck

The true gradient routed to block $\ell$ is

$$\frac{\partial \mathcal L}{\partial h_\ell} = \Big(\textstyle\prod_{k>\ell} J_k\Big)^{\!\top} e, \qquad e := \frac{\partial \mathcal L}{\partial h_L}$$

The central product $\prod_{k>\ell} J_k$ is the global backward chain: it must
cache every activation and runs strictly top-to-bottom. **LoCA's entire design is
about removing this product.**

### 3. Local target via a fixed feedback operator

Replace the input-dependent backward operator with a **fixed** matrix $F_\ell$ and
build a cheap per-token target from the top-layer error $e$:

$$\tau_\ell = h_\ell - \eta\, F_\ell\, e$$

The top error needs **only the LM head** — no backbone backward:

$$e = W_{\text{unembed}}^{\top}\big(\mathrm{softmax}(z) - \mathrm{onehot}(y)\big)$$

This is a descent direction **iff** the feedback is aligned with the true gradient,
$\alpha_\ell = \cos\angle(F_\ell e,\, g_\ell) > 0$. On a frozen LLM a *random*
$F_\ell$ gives $\alpha_\ell \approx 0$ (vanilla DFA fails). We instead fit a
**low-rank sketch** of the operator $e \mapsto g_\ell$ **once** on a small probe
batch, which yields $\alpha_\ell \approx 0.3\text{–}0.5$ and a steadily
decreasing loss.

### 4. Closed-form block solve

With the residual target $\rho_\ell = \tau_\ell - h_\ell^{\text{base}}$ and the
projected feature $p_\ell = A_\ell s_\ell$, each block is a ridge least-squares
problem with a unique closed-form minimiser:

$$B_\ell^\star = C_\ell\,(G_\ell + \lambda I)^{-1}, \qquad G_\ell = \sum_n p_\ell p_\ell^{\top}, \quad C_\ell = \sum_n \rho_\ell p_\ell^{\top}$$

$G_\ell\ (r\times r)$ and $C_\ell\ (d\times r)$ **accumulate additively over
batches**, so their size is independent of the token count — this is the source
of LoCA's flat-in-data memory. The system is solved in float64 for stability.

### 5. Outer iteration + early stopping

Targets depend on the current $B^{(t)}$, so we iterate $t = 1\dots T$ (Jacobi =
fully parallel, or Gauss–Seidel = sequential refresh). Global CE is non-monotone
in $T$ (the fixed $F_\ell$ drifts), so we select $T^\*$ by held-out CE and keep
the best snapshot — **returning the frozen model if no step beats it** (a
do-no-harm floor).

> **Scale-invariant variant.** Using a *relative* step
> $\eta_{\text{eff}} = \eta\,\lVert s\rVert / \lVert F_\ell e\rVert$ makes the
> correction a fixed fraction of the residual-stream norm — a single dimensionless
> $\eta$ then works across model sizes **and** architectures
> (`--loca-target-norm rms`).

### Algorithm

```text
Algorithm 1  LoCA (feedback variant)
─────────────────────────────────────────────────────────────────────
Input : frozen blocks {f_ℓ}, post-training data D, CE loss
Output: adapters {B_ℓ}  (with fixed {A_ℓ}, {F_ℓ})
Hyperparams: rank r, step η, ridge λ, outer iters T, mode∈{Jacobi,GS}

1  for ℓ = 1..L:  A_ℓ ~ N(0, 1/d) frozen;  B_ℓ ← 0
2  build {F_ℓ}: one-time low-rank sketch on a probe batch
3  for t = 1..T:
4      forward D through {f_ℓ, B_ℓ A_ℓ};  cache (s_ℓ, h_ℓ^base)   # 1 forward
5      e ← W_unembed^T (softmax(z) − onehot(y))                   # LM head only
6      for ℓ = 1..L:                                              # parallel (Jacobi)
7          p_ℓ ← A_ℓ s_ℓ ;   ρ_ℓ ← B_ℓ A_ℓ s_ℓ − η F_ℓ e
8          G_ℓ += Σ p_ℓ p_ℓ^T ;   C_ℓ += Σ ρ_ℓ p_ℓ^T            # streaming
9      for ℓ = 1..L:
10         B_ℓ ← C_ℓ (G_ℓ + λI)^{-1}                             # float64 solve
11     record held-out CE; keep best-T snapshot                  # early stopping
12 return best-T {B_ℓ}
─────────────────────────────────────────────────────────────────────
Per outer-iter cost ≈ 1 forward + L small solves.
No ∏ Jₖ, no cross-layer activation graph, no optimizer state.
```

---

## Repository layout

| Path | Contents |
|---|---|
| `src/adapters/` | `ResidualLoRA` (additive residual-stream low-rank) + model introspection |
| `src/loca/` | `hooks`, `top_error`, `feedback`, `closed_form`, `solver`, `diagnostics`, `runner` |
| `src/baselines/` | `lora_sft`, `mezo`, `full_sft`, `frozen` |
| `src/eval/` | `perplexity`, `format_acc`, `rank_acc`, `recovery`, `run_lm_eval` |
| `configs/` | base + per-method YAML (`loca_f`, `loca_d`, `baselines/*`) |
| `scripts/` | `phase0` (mechanism gate), `phase1` (quality matrix), `run_loca`, `run_baseline`, `sweep`, `make_figures` |
| `tests/` | unit tests + Phase-0 mechanism checks |

---

## Quick start

```bash
./setup_env.sh                 # or: pip install -r requirements.txt
python -m pytest tests/ -q     # unit + mechanism tests

# GO/NO-GO mechanism gate: alignment > 0 and CE decreases?
python scripts/phase0.py --model gpt2 --feedback sketch

# Quality matrix (frozen / LoRA / MeZO / LoCA) on a task
python scripts/phase1.py --model Qwen/Qwen2.5-0.5B --task boolq \
    --methods frozen lora mezo loca_f --loca-etas 0.003
```

See [`REPRODUCE.md`](REPRODUCE.md) for the full reproduction guide.

---

## Status

- **Mechanism (Phase 0) passes** on GPT-2 with `feedback=sketch`: alignment
  $\cos(F e, g) > 0$, global CE decreases, held-out CE beats frozen.
- `feedback=random` (vanilla DFA) shows $\approx 0$ alignment on a frozen LLM —
  the fitted **sketch** is the working feedback operator.
- Quality / efficiency / ablation studies across multiple model scales are run
  via the `scripts/` above.

> Research code accompanying an in-progress paper. Interfaces may change.
