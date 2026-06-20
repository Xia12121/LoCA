"""Wall-clock + peak-memory profiling (§7, C3).

CPU peak RSS via resource.getrusage; CUDA peak via torch.cuda.max_memory_allocated.
The headline C3 claim depends on these numbers being honestly collected on the
SAME machine for every method, so keep this dead simple and side-effect free.
"""
from __future__ import annotations

import contextlib
import platform
import resource
import time
from dataclasses import dataclass

import torch


def _rss_mb() -> float:
    """Resident set size in MB. ru_maxrss is bytes on macOS, kB on Linux."""
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return val / (1024 * 1024)
    return val / 1024


@dataclass
class ProfileResult:
    wall_clock_s: float
    peak_mem_mb: float          # process peak RSS (CPU view, always valid)
    cuda_peak_mb: float = 0.0   # CUDA peak allocated, 0 on CPU


@contextlib.contextmanager
def profile_block(device: str = "cpu"):
    """Context manager yielding a mutable ProfileResult, filled on exit.

    Usage:
        with profile_block(device) as p:
            ... work ...
        print(p.wall_clock_s, p.peak_mem_mb)
    """
    res = ProfileResult(0.0, 0.0, 0.0)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield res
    finally:
        if use_cuda:
            torch.cuda.synchronize()
            res.cuda_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        res.wall_clock_s = time.perf_counter() - t0
        res.peak_mem_mb = _rss_mb()


class Timer:
    """Lightweight named-section timer for breaking down where time goes."""

    def __init__(self) -> None:
        self.sections: dict[str, float] = {}
        self._stack: list[tuple[str, float]] = []

    @contextlib.contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.sections[name] = self.sections.get(name, 0.0) + dt

    def summary(self) -> dict[str, float]:
        return dict(self.sections)
