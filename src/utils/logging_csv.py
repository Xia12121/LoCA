"""Append-only CSV result logging (§7).

Every experiment row records enough to reproduce + compare:
  method, model, task, seed, metric, value, wall_clock_s, peak_mem_mb, config_hash, extra(json)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import Any

_LOCK = threading.Lock()

FIELDS = [
    "method",
    "model",
    "task",
    "seed",
    "metric",
    "value",
    "wall_clock_s",
    "peak_mem_mb",
    "config_hash",
    "extra",
]


@dataclass
class ResultRow:
    method: str
    model: str
    task: str
    seed: int
    metric: str
    value: float
    wall_clock_s: float = 0.0
    peak_mem_mb: float = 0.0
    config_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_csv_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["extra"] = json.dumps(d["extra"], sort_keys=True)
        return d


def config_hash(config: dict) -> str:
    """Stable short hash of a config dict for grouping rows."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def append_row(csv_path: str, row: ResultRow) -> None:
    """Thread-safe append; writes header if file is new."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with _LOCK:
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row.to_csv_dict())


def append_rows(csv_path: str, rows: list[ResultRow]) -> None:
    for r in rows:
        append_row(csv_path, r)
