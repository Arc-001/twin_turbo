"""Live estimator vs offline replay must agree exactly on real engines.
Requires `python scripts/run_experiments.py --dataset FD001` to have run."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts.simulate_engine import simulate
from src.data_pipeline.prepare import load_raw
from src.decision.hybrid import EscalationThresholds
from src.decision.replay import hybrid_lower_bound
from src.experiments.traces import build_traces

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "artifacts" / "models" / "FD001_bundle.pkl"


@pytest.mark.skipif(not BUNDLE.exists(), reason="run scripts/run_experiments.py first")
@pytest.mark.parametrize("unit", [1, 34, 81])
def test_live_estimator_matches_replay(unit):
    with open(BUNDLE, "rb") as f:
        bundle = pickle.load(f)
    _, test_raw = load_raw("FD001", REPO / "CMAPSSData")
    df = bundle.data.regime_model.normalize(test_raw[test_raw["unit"] == unit])

    live = simulate(bundle, df, seed=unit)
    trace = build_traces(bundle, df)
    lo, synced, _ = hybrid_lower_bound(trace, EscalationThresholds())

    np.testing.assert_allclose(live["edge_rul"], trace["edge"], rtol=1e-9)
    np.testing.assert_array_equal(live["twin_synced"].to_numpy(), synced)
    np.testing.assert_allclose(live["final_lo"], lo, rtol=1e-9, atol=1e-9)
