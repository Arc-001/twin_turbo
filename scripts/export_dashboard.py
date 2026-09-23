#!/usr/bin/env python3
"""Export real model traces + results for the offline demo dashboard.

Writes dashboard/demo-data.js, which defines `window.TWIN_DATA`. The page loads
it with a plain <script> tag, so the dashboard opens straight from disk with no
server. Every number the dashboard shows or recomputes comes from here.

Numeric series are stored as delta-encoded integers (first value, then
cycle-to-cycle differences), scaled by 100 for RUL, 1000 for the health index
and 100 for normalized sensors; the page's decoder reverses this. Decision
logic in the page runs on exactly these decoded values.

Usage:
    python scripts/export_dashboard.py
"""

import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_WINDOW_SIZE, VARIANTS
from src.data_pipeline.windowing import build_windows
from src.decision.hybrid import EscalationThresholds
from src.decision.policy import DecisionThresholds
from src.decision.replay import Z90, Costs
from src.eval.metrics import rmse
from src.models.features import summarize_windows

# Sensor names and meanings from Saxena et al. (2008), Table 2.
SENSOR_INFO = {
    "s1": ("T2", "Fan inlet temperature"), "s2": ("T24", "LPC outlet temperature"),
    "s3": ("T30", "HPC outlet temperature"), "s4": ("T50", "LPT outlet temperature"),
    "s5": ("P2", "Fan inlet pressure"), "s6": ("P15", "Bypass-duct pressure"),
    "s7": ("P30", "HPC outlet pressure"), "s8": ("Nf", "Physical fan speed"),
    "s9": ("Nc", "Physical core speed"), "s10": ("epr", "Engine pressure ratio"),
    "s11": ("Ps30", "HPC outlet static pressure"), "s12": ("phi", "Fuel flow / Ps30"),
    "s13": ("NRf", "Corrected fan speed"), "s14": ("NRc", "Corrected core speed"),
    "s15": ("BPR", "Bypass ratio"), "s16": ("farB", "Burner fuel-air ratio"),
    "s17": ("htBleed", "Bleed enthalpy"), "s18": ("Nf_dmd", "Demanded fan speed"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed"), "s20": ("W31", "HPT coolant bleed"),
    "s21": ("W32", "LPT coolant bleed"),
}
N_SENSOR_STRIP = 3


def enc(a, scale: int) -> list[int]:
    q = np.round(np.asarray(a, dtype=float) * scale).astype(int)
    return np.concatenate([q[:1], np.diff(q)]).tolist()


def i10(a) -> list[int]:
    return enc(a, 100)


def i1000(a) -> list[int]:
    return enc(a, 1000)


def edge_sensor_importance(bundle) -> dict[str, float]:
    """Permutation importance grouped by sensor: shuffle all four summary
    features of one sensor together on the official test windows, measure the
    RMSE increase."""
    data = bundle.data
    X, y, _ = build_windows(data.test, data.feature_cols, DEFAULT_WINDOW_SIZE, last_only=True)
    F = summarize_windows(X)
    base = rmse(y, bundle.edge.predict(F))
    n = len(data.feature_cols)
    rng = np.random.default_rng(0)
    out = {}
    for j, col in enumerate(data.feature_cols):
        deltas = []
        for _ in range(5):
            Fp = F.copy()
            perm = rng.permutation(len(F))
            for k in range(4):
                Fp[:, k * n + j] = F[perm, k * n + j]
            deltas.append(rmse(y, bundle.edge.predict(Fp)) - base)
        out[col] = float(np.mean(deltas))
    return out


def export_dataset(ds: str, art: Path, results: dict) -> dict:
    meta = json.loads((art / "traces" / f"{ds}_meta.json").read_text())
    cv = pd.read_pickle(art / "traces" / f"{ds}_cv.pkl")
    with open(art / "models" / f"{ds}_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    data = bundle.data
    variant = VARIANTS[ds]

    hi_w = meta["hi_weights"]
    strip = sorted(hi_w, key=lambda s: -abs(hi_w[s]))[:N_SENSOR_STRIP]

    # normalized sensor values for the strip, from the same fold-free protocol-A normalizer
    train_norm = data.train.set_index(["unit", "cycle"])

    ages = {int(k): int(v) for k, v in meta["ages_by_fold"].items()}
    engines = []
    for unit, g in cv.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        sens = train_norm.loc[int(unit)].loc[g["cycle"].to_numpy(), strip]
        engines.append({
            "u": int(unit), "fold": int(g["fold"].iloc[0]), "life": int(g["cycle"].iloc[-1]),
            "age": ages[int(g["fold"].iloc[0])],
            "edge": i10(g["edge"]), "elo": i10(g["edge_lo"]), "ehi": i10(g["edge_hi"]),
            "t05": i10(g["twin_q05"]), "t50": i10(g["twin_q50"]), "t95": i10(g["twin_q95"]),
            "lstm": i10(g["lstm"]), "llo": i10(g["lstm_mc_mean"] - Z90 * g["lstm_mc_std"]),
            "hi": i1000(g["hi"]),
            "sens": [enc(sens[c], 100) for c in strip],
        })

    d = results["datasets"][ds]
    imp = edge_sensor_importance(bundle)
    return {
        "name": ds, "regimes": variant.n_regimes, "faults": variant.n_fault_modes,
        "nFeatures": len(data.feature_cols), "featureCols": data.feature_cols,
        "hiWeights": hi_w, "edgeImportance": imp, "strip": strip,
        "prior": meta["prior"], "conformal": meta["conformal"], "ages": ages,
        "engines": engines,
        "results": {
            "benchmark": d["benchmark"], "cvAccuracy": d["cv_accuracy"],
            "decisions": {k: d["decisions"][k] for k in ("default", "sweep", "triggers", "resync", "fusion", "safe_operating_point")},
            "latency": d["latency"], "nCvUnits": d["n_cv_units"],
        },
    }


def main() -> None:
    art = REPO_ROOT / "artifacts"
    results = json.loads((art / "results" / "results.json").read_text())
    esc, th, costs = EscalationThresholds(), DecisionThresholds(), Costs()
    payload = {
        "generated": date.today().isoformat(),
        "scale": {"rul": 100, "hi": 1000, "sens": 100},
        "defaults": {
            "safe": th.safe, "watch": th.watch, "ground": th.ground,
            "watchRul": esc.watch_rul, "deltaDrop": esc.delta_drop, "resync": esc.resync_period,
            "costPreventive": costs.preventive, "costFailure": costs.failure,
        },
        "sensorInfo": SENSOR_INFO,
        "literature": results["literature"],
        "datasets": {},
    }
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        payload["datasets"][ds] = export_dataset(ds, art, results)
        print(f"[{ds}] exported {len(payload['datasets'][ds]['engines'])} engines")
    out = REPO_ROOT / "dashboard" / "demo-data.js"
    body = json.dumps(payload, separators=(",", ":"))
    out.write_text("/* generated by scripts/export_dashboard.py -- do not edit */\nwindow.TWIN_DATA=" + body + ";\n")
    print(f"-> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
