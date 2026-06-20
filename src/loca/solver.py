"""Outer block-coordinate iteration — Algorithm 1 (§4.7, §5).

Per outer iteration t:
  1. (one) cached forward over the data -> per-block (s, h0) and top hidden h_L
  2. e = top_layer_error(h_L, labels)                       # only through LM head
  3. for each block l:  p = A_l s ;  rho = correction_l(s) - eta F_l e   # eq. (§4.6)
                        accumulate streaming Gram (G_l, C_l)
  4. solve B_l = C_l (G_l + lam I)^{-1}  and write back

Modes:
  jacobi       : one forward, all blocks accumulated together, all solved -> parallel
  gauss_seidel : per block, re-forward with updated upstream, solve, advance

Kill criteria (§8) are checked inline and raise KillSignal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..adapters.model_utils import get_handles
from ..adapters.residual_lora import AdaptedBlock
from .closed_form import GramAccumulator
from .hooks import cached_forward
from .top_error import top_layer_error


class KillSignal(Exception):
    """Raised when a §8 KILL_CRITERIA condition fires; caught by the runner."""

    def __init__(self, key: str, message: str):
        super().__init__(f"[{key}] {message}")
        self.key = key


@dataclass
class OuterMetrics:
    t: int
    global_ce: float            # mean CE over label tokens (health signal)
    local_residual: float       # mean_l mean_tok 0.5||rho||^2 before solve
    b_norm_max: float           # max ||B_l||_F (explosion guard)
    held_out: float = float("nan")  # held-out score if early-stopping is on


@dataclass
class LoCAConfig:
    eta: float = 0.05
    lam: float = 0.1
    T: int = 5
    mode: str = "jacobi"        # jacobi | gauss_seidel
    variant: str = "F"          # F | D
    target_norm: str = "none"   # none | rms (scale-invariant relative step)
    check_monotone: bool = True


def _global_ce(h_L: Tensor, labels: Tensor, lm_head) -> float:
    with torch.no_grad():
        z = h_L.to(lm_head.weight.dtype) @ lm_head.weight.t()
        b = getattr(lm_head, "bias", None)
        if b is not None:
            z = z + b
        ce = torch.nn.functional.cross_entropy(z.float(), labels, reduction="mean")
    return float(ce)


class LoCASolver:
    """Drives Algorithm 1 over a list of batches."""

    def __init__(self, model, blocks: list[AdaptedBlock], feedback: list[Tensor], cfg: LoCAConfig):
        self.model = model
        self.blocks = blocks
        self.F = feedback
        self.cfg = cfg
        self.handles = get_handles(model)
        self.device = next(model.parameters()).device
        self.history: list[OuterMetrics] = []

    # ---- per-block residual target (§4.6) ------------------------------- #
    def _rho(self, block: AdaptedBlock, s: Tensor, Fe: Tensor) -> Tensor:
        """rho_l = correction_l^{(t)}(s) - eta F_l e   (eq. in §4.6).

        correction_l = B_l^{(t)} A_l s (current adapter output); on t=1 B=0 so
        rho = -eta F_l e.
        """
        corr = block.adapter.correction(s).to(torch.float32)     # (n_tok, d)
        if self.cfg.target_norm == "rms":
            # Scale-invariant target: eta becomes a RELATIVE step (a fraction of the
            # residual-stream norm), so ONE eta works across model sizes. We rescale
            # the feedback so eta_eff = eta * mean||s|| / mean||Fe||  ->  the correction
            # magnitude is ~ eta * ||s|| regardless of how F / e happen to be scaled.
            s_rms = s.to(torch.float32).norm(dim=-1).mean().clamp_min(1e-8)
            fe_rms = Fe.norm(dim=-1).mean().clamp_min(1e-8)
            return corr - (self.cfg.eta * (s_rms / fe_rms)) * Fe
        return corr - self.cfg.eta * Fe

    def _feedback_signal(self, l: int, e: Tensor) -> Tensor:
        """F_l e  for all tokens. e: (n_tok, d_top) -> (n_tok, d)."""
        return e @ self.F[l].to(device=e.device, dtype=e.dtype).t()

    # ---- one Jacobi outer iteration ------------------------------------- #
    def _outer_jacobi(self, batches: list[dict]) -> OuterMetrics:
        L = len(self.blocks)
        accs = [GramAccumulator(self.handles.hidden_size, self.blocks[0].adapter.r, device=self.device) for _ in range(L)]
        ce_sum, ce_tok = 0.0, 0
        resid_sum, resid_cnt = 0.0, 0

        for batch in batches:
            pmask = batch["predict_mask"].to(self.device)
            caches, h_L = cached_forward(
                self.model, self.blocks,
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                pmask,
            )
            labels = batch["targets"].to(self.device)[pmask]   # next-token targets at predict positions
            e = top_layer_error(h_L, labels, self.handles.lm_head, reduction="sum")  # (n_tok, d_top)
            ce_sum += _global_ce(h_L, labels, self.handles.lm_head) * h_L.shape[0]
            ce_tok += h_L.shape[0]

            for l, (cache, block) in enumerate(zip(caches, self.blocks)):
                s = cache.s.to(self.device)
                p = block.adapter.project(s).to(torch.float32)        # (n_tok, r)
                Fe = self._feedback_signal(l, e)
                rho = self._rho(block, s, Fe)                          # (n_tok, d)
                accs[l].add(p, rho)
                resid_sum += 0.5 * float((rho * rho).sum())
                resid_cnt += rho.shape[0]

        # solve all blocks (parallel-safe: independent)
        b_norm_max = 0.0
        for l, block in enumerate(self.blocks):
            B = accs[l].solve(self.cfg.lam)
            self._check_finite(B, l)
            block.adapter.set_B(B)
            b_norm_max = max(b_norm_max, float(B.norm()))

        return OuterMetrics(
            t=0,
            global_ce=ce_sum / max(ce_tok, 1),
            local_residual=resid_sum / max(resid_cnt, 1),
            b_norm_max=b_norm_max,
        )

    # ---- one Gauss-Seidel outer iteration ------------------------------- #
    def _outer_gauss_seidel(self, batches: list[dict]) -> OuterMetrics:
        """Solve block-by-block; each block sees upstream blocks already updated.

        Implemented straightforwardly: for each l, re-run the cached forward (so
        upstream adapters are current), accumulate only block l, solve, advance.
        Costs L forwards/outer-iter — fine for small models; documented in paper.
        """
        L = len(self.blocks)
        ce_last, resid_sum, resid_cnt, b_norm_max = 0.0, 0.0, 0, 0.0
        for l in range(L):
            acc = GramAccumulator(self.handles.hidden_size, self.blocks[l].adapter.r, device=self.device)
            ce_sum, ce_tok = 0.0, 0
            for batch in batches:
                pmask = batch["predict_mask"].to(self.device)
                caches, h_L = cached_forward(
                    self.model, self.blocks,
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    pmask,
                )
                labels = batch["targets"].to(self.device)[pmask]
                e = top_layer_error(h_L, labels, self.handles.lm_head, reduction="sum")
                ce_sum += _global_ce(h_L, labels, self.handles.lm_head) * h_L.shape[0]
                ce_tok += h_L.shape[0]
                s = caches[l].s.to(self.device)
                p = self.blocks[l].adapter.project(s).to(torch.float32)
                Fe = self._feedback_signal(l, e)
                rho = self._rho(self.blocks[l], s, Fe)
                acc.add(p, rho)
                resid_sum += 0.5 * float((rho * rho).sum())
                resid_cnt += rho.shape[0]
            B = acc.solve(self.cfg.lam)
            self._check_finite(B, l)
            self.blocks[l].adapter.set_B(B)
            b_norm_max = max(b_norm_max, float(B.norm()))
            ce_last = ce_sum / max(ce_tok, 1)
        return OuterMetrics(t=0, global_ce=ce_last,
                            local_residual=resid_sum / max(resid_cnt, 1),
                            b_norm_max=b_norm_max)

    def _check_finite(self, B: Tensor, l: int) -> None:
        if not torch.isfinite(B).all():
            raise KillSignal("nan_or_explode", f"non-finite B at block {l}; raise lam / lower eta")

    # ---- one outer step (public, for diagnostics-driven loops) ---------- #
    def outer_step(self, batches: list[dict]) -> OuterMetrics:
        if self.cfg.mode == "jacobi":
            m = self._outer_jacobi(batches)
        elif self.cfg.mode == "gauss_seidel":
            m = self._outer_gauss_seidel(batches)
        else:
            raise ValueError(f"unknown mode {self.cfg.mode}")
        if m.b_norm_max > 1e6 or not torch.isfinite(torch.tensor(m.global_ce)):
            raise KillSignal("nan_or_explode", f"explosion: ce={m.global_ce}, |B|={m.b_norm_max}")
        return m

    # ---- best-B snapshot / restore (early stopping over T) -------------- #
    @torch.no_grad()
    def _snapshot_B(self) -> list[Tensor]:
        return [b.adapter.B.detach().clone() for b in self.blocks]

    @torch.no_grad()
    def _restore_B(self, snap: list[Tensor]) -> None:
        for b, B in zip(self.blocks, snap):
            b.adapter.set_B(B)

    # ---- public driver -------------------------------------------------- #
    def fit(self, batches: list[dict], on_iter=None, eval_fn=None) -> list[OuterMetrics]:
        """Run T outer iterations. `on_iter(t, metrics, solver)` fires after each.

        If `eval_fn` is given, it returns a held-out score to MINIMIZE (e.g. CE);
        the best outer iterate is snapshotted and restored at the end. Because the
        fixed feedback F drifts as the model moves, global CE is non-monotone in T,
        so selecting T by held-out score (a swept hyperparameter) is the intended
        model-selection step, not a hack.
        """
        prev_ce = float("inf")
        best_score, best_snap, best_t = float("inf"), None, 0
        if eval_fn is not None:
            best_score, best_snap, best_t = eval_fn(), self._snapshot_B(), 0  # frozen baseline
        for t in range(1, self.cfg.T + 1):
            m = self.outer_step(batches)
            m.t = t
            self.history.append(m)
            if self.cfg.check_monotone and t >= 2 and m.global_ce > prev_ce + 1e-4:
                self._monotone_violation = (t, prev_ce, m.global_ce)
            prev_ce = m.global_ce
            if eval_fn is not None:
                score = eval_fn()
                m.held_out = score
                if score < best_score:
                    best_score, best_snap, best_t = score, self._snapshot_B(), t
            if on_iter is not None:
                on_iter(t, m, self)
        if best_snap is not None:
            self._restore_B(best_snap)
            self.best_t, self.best_score = best_t, best_score
        return self.history
