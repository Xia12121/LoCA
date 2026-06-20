"""Block-wise closed-form ridge solve with streaming Gram accumulation (§4.6).

For block l, with per-token projection p = A_l s (in R^r) and residual target
rho = tau_l - h_l^base (in R^d), minimize  sum ||B p - rho||^2 + lam ||B||_F^2.

Closed form (eq. 8):
    B_l = ( sum_tok rho p^T ) ( sum_tok p p^T + lam I_r )^{-1}
        = C_l (G_l + lam I_r)^{-1}

Streaming: accumulate only the two small matrices, independent of token count:
    G_l += sum_batch p p^T     (r, r)
    C_l += sum_batch rho p^T   (d, r)

Solve via torch.linalg.solve (NOT explicit inverse), in float64, then cast back.
This O(L*(r^2 + d*r)) state is why CPU RAM stays flat in #tokens.
"""
from __future__ import annotations

import torch
from torch import Tensor


class GramAccumulator:
    """Streaming (G, C) for one block. Add label-token batches, then solve once."""

    def __init__(self, d_model: int, r: int, device: str = "cpu"):
        self.d = d_model
        self.r = r
        self.G = torch.zeros(r, r, dtype=torch.float64, device=device)
        self.C = torch.zeros(d_model, r, dtype=torch.float64, device=device)
        self.n = 0

    @torch.no_grad()
    def add(self, P_proj: Tensor, Rho: Tensor) -> None:
        """P_proj: (N, r) = p per token ; Rho: (N, d) = rho per token."""
        P = P_proj.to(torch.float64)
        R = Rho.to(torch.float64)
        self.G += P.t() @ P             # (r, r)
        self.C += R.t() @ P             # (d, r)
        self.n += P.shape[0]

    @torch.no_grad()
    def solve(self, lam: float) -> Tensor:
        """B = C (G + lam I)^{-1}, computed as B^T = solve(G+lam I, C^T)."""
        return solve_block(self.G, self.C, lam)


@torch.no_grad()
def accumulate_gram(P_proj: Tensor, Rho: Tensor, G: Tensor, C: Tensor) -> None:
    """Functional form of GramAccumulator.add (in-place on G, C)."""
    P = P_proj.to(G.dtype)
    R = Rho.to(C.dtype)
    G += P.t() @ P
    C += R.t() @ P


@torch.no_grad()
def solve_block(G: Tensor, C: Tensor, lam: float) -> Tensor:
    """Return B = C @ (G + lam I)^{-1} via torch.linalg.solve, float64 internally.

    G: (r, r), C: (d, r) -> B: (d, r), returned in float32.
    Solve (G + lam I) X = C^T  =>  X = (G+lam I)^{-1} C^T  (r, d), then B = X^T.
    """
    r = G.shape[0]
    Gd = G.to(torch.float64)
    Cd = C.to(torch.float64)
    A = Gd + lam * torch.eye(r, dtype=torch.float64, device=Gd.device)
    X = torch.linalg.solve(A, Cd.t())     # (r, d)
    B = X.t().contiguous()                # (d, r)
    return B.to(torch.float32)
