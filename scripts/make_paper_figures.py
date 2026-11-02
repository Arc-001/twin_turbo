#!/usr/bin/env python3
"""Figures for the journal manuscript (report/ijit/figures/), drawn from cached
traces and results.json so they match the reported numbers exactly.

Usage:
    python scripts/make_paper_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.decision.hybrid import EscalationThresholds
from src.decision.replay import hybrid_lower_bound

ART = REPO_ROOT / "artifacts"
OUT = REPO_ROOT / "report" / "ijit" / "figures"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
COLORS = {"edge": "#2a6fdb", "twin": "#d1495b", "lstm": "#8a8a8a", "hybrid": "#1b998b", "true": "#222222"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})


def engine_replay(ds: str = "FD001", unit: int = 14) -> None:
    cv = pd.read_pickle(ART / "traces" / f"{ds}_cv.pkl")
    g = cv[cv["unit"] == unit].sort_values("cycle")
    lo, synced, _ = hybrid_lower_bound(g, EscalationThresholds())
    c = g["cycle"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.fill_between(c, g["edge_lo"], g["edge_hi"], color=COLORS["edge"], alpha=0.15, lw=0, label="Edge 90% conformal band")
    ax.fill_between(c, np.minimum(g["twin_q05"], 125), np.minimum(g["twin_q95"], 125), color=COLORS["twin"], alpha=0.15,
                    lw=0, label="Twin 90% posterior band")
    ax.plot(c, g["rul"], color=COLORS["true"], lw=1.2, label="True RUL (clipped at 125)")
    ax.plot(c, g["edge"], color=COLORS["edge"], lw=0.8, label="Edge estimate")
    ax.plot(c, np.minimum(g["twin_q50"], 125), color=COLORS["twin"], lw=0.8, label="Twin posterior median")
    ax.plot(c, lo, color=COLORS["hybrid"], lw=1.0, ls="--", label="Hybrid lower bound")
    ax.plot(c[synced], np.full(synced.sum(), -6), "|", color=COLORS["hybrid"], ms=4, label="Twin synchronization")
    for y, name in [(60, "WATCH"), (30, "SCHEDULE"), (10, "GROUND")]:
        ax.axhline(y, color="0.6", lw=0.5, ls=":")
        ax.text(c[0] + 1, y + 1.5, name, color="0.45", fontsize=6)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_ylim(-12, 150)
    ax.set_xlim(c[0], c[-1])
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "engine_replay.pdf")
    plt.close(fig)


def coverage_by_horizon(results: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.1), sharey=True)
    series = [("picp_edge_conformal", "Edge conformal", COLORS["edge"], "o"),
              ("picp_twin_posterior", "Twin posterior", COLORS["twin"], "s"),
              ("picp_lstm_mc_dropout", "LSTM MC dropout", COLORS["lstm"], "^")]
    for ax, ds in zip(axes, DATASETS):
        rows = results["datasets"][ds]["cv_accuracy"]["by_horizon"]
        x = [(r["lo"] + r["hi"]) / 2 for r in rows]
        for key, label, col, m in series:
            ax.plot(x, [r[key] for r in rows], color=col, marker=m, ms=3, lw=1, label=label)
        ax.axhline(0.9, color="0.5", lw=0.6, ls="--")
        ax.set_title(ds)
        ax.set_xlabel("True RUL (cycles)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Empirical coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "coverage_horizon.pdf")
    plt.close(fig)


def threshold_sweep(results: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.2), sharey=False)
    policies = [("edge_point", "Edge point", COLORS["edge"], ":"), ("edge_conformal", "Edge conformal", COLORS["edge"], "-"),
                ("lstm_mc", "LSTM MC dropout", COLORS["lstm"], "-"), ("twin_always", "Twin only", COLORS["twin"], "-"),
                ("hybrid", "Hybrid", COLORS["hybrid"], "-")]
    for ax, ds in zip(axes, DATASETS):
        sw = pd.DataFrame(results["datasets"][ds]["decisions"]["sweep"])
        for p, label, col, ls in policies:
            s = sw[sw["policy"] == p].sort_values("ground")
            ax.plot(s["ground"], s["failures"], color=col, ls=ls, lw=1, marker="o", ms=2.5, label=label)
        ax.set_title(ds)
        ax.set_xlabel("Grounding threshold $\\tau_g$")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(-0.2, 400)
    axes[0].set_ylabel("Engines failed")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "threshold_sweep.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads((ART / "results" / "results.json").read_text())
    engine_replay()
    coverage_by_horizon(results)
    threshold_sweep(results)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
