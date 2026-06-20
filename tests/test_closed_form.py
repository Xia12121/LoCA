"""E0.2 — closed-form ridge solve vs numpy reference (acceptance: rtol < 1e-5)."""
import numpy as np
import torch

from src.loca.closed_form import GramAccumulator, solve_block


def _numpy_ridge(P, Rho, lam):
    # B = (Rho^T P)(P^T P + lam I)^-1 ; P:(N,r) Rho:(N,d)
    r = P.shape[1]
    G = P.T @ P
    C = Rho.T @ P
    return C @ np.linalg.inv(G + lam * np.eye(r))


def test_solve_block_matches_numpy_ridge():
    torch.manual_seed(0)
    N, r, d, lam = 200, 8, 16, 0.3
    P = torch.randn(N, r, dtype=torch.float64)
    Rho = torch.randn(N, d, dtype=torch.float64)
    G = P.t() @ P
    C = Rho.t() @ P
    B = solve_block(G, C, lam).double().numpy()
    B_ref = _numpy_ridge(P.numpy(), Rho.numpy(), lam)
    assert np.allclose(B, B_ref, rtol=1e-5, atol=1e-8), np.abs(B - B_ref).max()


def test_streaming_accumulator_equivalence():
    torch.manual_seed(1)
    N, r, d, lam = 500, 16, 32, 0.1
    P = torch.randn(N, r)
    Rho = torch.randn(N, d)
    # full
    acc_full = GramAccumulator(d, r)
    acc_full.add(P, Rho)
    B_full = acc_full.solve(lam)
    # streamed in chunks
    acc_str = GramAccumulator(d, r)
    for i in range(0, N, 37):
        acc_str.add(P[i:i + 37], Rho[i:i + 37])
    B_str = acc_str.solve(lam)
    assert torch.allclose(B_full, B_str, rtol=1e-5, atol=1e-6)


def test_exact_recovery_low_rank_target():
    """If rho = B_true p exactly, ridge with tiny lam recovers B_true."""
    torch.manual_seed(2)
    N, r, d = 1000, 8, 12
    P = torch.randn(N, r)
    B_true = torch.randn(d, r)
    Rho = P @ B_true.t()
    acc = GramAccumulator(d, r)
    acc.add(P, Rho)
    B = acc.solve(1e-8)
    assert torch.allclose(B, B_true, rtol=1e-3, atol=1e-3), (B - B_true).abs().max()
