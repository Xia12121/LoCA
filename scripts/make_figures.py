"""Regenerate paper figures from outputs/results.csv (§7, T12).

  python scripts/make_figures.py --csv outputs/results.csv --out outputs/figures

Produces whatever the CSV supports:
  - recovery bar per task/method (C1/C2)
  - CPU wall-clock vs model size (C3 headline)
  - alignment angle vs outer iteration (C4a) [from diagnostics CSV if present]
  - sweep curves (eta/lam/T/r)
Figures degrade gracefully: a panel is skipped (with a note) if its data is absent.
"""
from __future__ import annotations

import argparse
import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _parse_extra(df):
    def safe(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) and x.strip() else {}
        except Exception:
            return {}
    extra = df["extra"].map(safe)
    for k in set().union(*[d.keys() for d in extra]) if len(extra) else []:
        df[f"x_{k}"] = extra.map(lambda d: d.get(k))
    return df


def fig_recovery(df, out):
    sub = df[df["metric"].isin(["ce", "perplexity"])]
    if sub.empty or "frozen" not in set(sub["method"]) or "lora" not in set(sub["method"]):
        print("[fig] recovery: need frozen + lora rows; skipping")
        return
    metric = "ce"
    piv = (sub[sub["metric"] == metric]
           .groupby(["task", "method"])["value"].mean().reset_index())
    tasks = sorted(piv["task"].unique())
    methods = [m for m in ["frozen", "mezo", "loca_d", "loca_f", "lora", "full_sft"] if m in set(piv["method"])]
    fig, ax = plt.subplots(figsize=(1.6 * len(tasks) + 3, 4))
    rows = []
    for task in tasks:
        d = piv[piv["task"] == task].set_index("method")["value"]
        if "frozen" not in d or "lora" not in d:
            continue
        fr, lo = d["frozen"], d["lora"]
        for m in methods:
            if m in d and abs(fr - lo) > 1e-9:
                rec = (fr - d[m]) / (fr - lo)   # lower CE is better
                rows.append((task, m, rec))
    if not rows:
        print("[fig] recovery: no comparable rows; skipping")
        return
    rdf = pd.DataFrame(rows, columns=["task", "method", "recovery"])
    for i, m in enumerate(methods):
        vals = [rdf[(rdf.task == t) & (rdf.method == m)]["recovery"].mean() for t in tasks]
        ax.bar([x + i * 0.13 for x in range(len(tasks))], vals, width=0.13, label=m)
    ax.axhline(0.85, ls="--", c="k", lw=1, label="C1 target 0.85")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("recovery (CE)")
    ax.set_title("Recovery of LoRA gain (C1/C2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out, "recovery.png")
    fig.savefig(p, dpi=150)
    print(f"[fig] wrote {p}")


def fig_cpu_scaling(df, out):
    sub = df[(df["metric"] == "ce")].copy()
    if sub.empty or sub["wall_clock_s"].fillna(0).max() <= 0:
        print("[fig] cpu_scaling: no wall-clock data; skipping")
        return
    g = sub.groupby(["model", "method"])["wall_clock_s"].mean().reset_index()
    if g["model"].nunique() < 2:
        print("[fig] cpu_scaling: need >=2 model sizes; skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in sorted(g["method"].unique()):
        d = g[g["method"] == m].sort_values("model")
        ax.plot(d["model"], d["wall_clock_s"], marker="o", label=m)
    ax.set_yscale("log")
    ax.set_ylabel("wall-clock / run (s, log)")
    ax.set_xlabel("model")
    ax.set_title("CPU wall-clock vs model size (C3)")
    ax.legend(fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    p = os.path.join(out, "cpu_scaling.png")
    fig.savefig(p, dpi=150)
    print(f"[fig] wrote {p}")


def fig_sweep(df, out):
    df = _parse_extra(df.copy())
    if "x_swept_param" not in df.columns:
        print("[fig] sweep: no swept rows; skipping")
        return
    sub = df[(df["metric"] == "ce") & df["x_swept_param"].notna()]
    for param in sub["x_swept_param"].dropna().unique():
        d = sub[sub["x_swept_param"] == param]
        g = d.groupby("x_swept_value")["value"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(g["x_swept_value"].astype(float), g["mean"], yerr=g["std"].fillna(0), marker="o")
        ax.set_xlabel(param)
        ax.set_ylabel("held-out CE")
        ax.set_title(f"Sweep: {param}")
        fig.tight_layout()
        p = os.path.join(out, f"sweep_{param.replace('.', '_')}.png")
        fig.savefig(p, dpi=150)
        print(f"[fig] wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/results.csv")
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(args.csv):
        print(f"[fig] no csv at {args.csv}")
        return
    df = pd.read_csv(args.csv)
    fig_recovery(df, args.out)
    fig_cpu_scaling(df, args.out)
    fig_sweep(df, args.out)
    print("[fig] done")


if __name__ == "__main__":
    main()
