"""Check 17: calibration smoke test on a small setup (the full numbers come from E3).

Six null markets (20 instruments, 500 bars, development seeds) with the E3 grid restricted to 24
trials: the selected trial's per-bar Sharpe ratio is within a factor of 2 of SR0 on average.
A planted market with a strong signal (ic 0.15, 1000 bars): the selected trial uses signal_x and
its DSR is above 0.95 (with 500 bars it was 0.84: many trials share the signal, which raises the
variance of the trials' Sharpe ratios and so SR0)."""

import numpy as np
import pytest

from quant_research_engine.costs import zero_costs
from quant_research_engine.experiments import e3
from quant_research_engine.experiments.markets import market
from quant_research_engine.experiments.runner import load_config
from quant_research_engine.inference.registry import TrialRegistry


def _cfg():
    cfg = load_config("e3", "quick")
    grid = cfg["grid"]
    grid = {
        **grid,
        "signals": [{"feature": s["feature"], "params": s["params"][:2]} for s in grid["signals"]],
        "rebalance": [1, 5],
    }
    return {**cfg, "grid": grid, "n_instruments": 20, "n_bars": 500, "first_decision_bar": 259}


def _select(R):
    reg = TrialRegistry("smoke")
    for j in range(R.shape[1]):
        reg.add(str(j), {"j": j}, R[:, j])
    srs = np.asarray([r[3] for r in reg.rows])
    sel = int(np.argmax(srs))
    return sel, srs[sel], reg.sr0(), reg.dsr(sel)


@pytest.mark.slow
def test_best_null_trial_is_near_sr0_and_a_strong_signal_is_selected():
    cfg = _cfg()
    trials = e3.trial_list(cfg)
    assert len(trials) == 32
    ratios = []
    for seed in range(1, 7):
        st, _ = market("smoke", "null_e3", seed, n_instruments=20, n_bars=500)
        R, _ = e3.returns_matrix(st, cfg, trials, zero_costs())
        _, sr, sr0, _ = _select(R)
        ratios.append(sr / sr0)
    assert 0.5 <= float(np.mean(ratios)) <= 2.0, ratios
    cfg = {**cfg, "n_bars": 1000}
    st, _ = market("smoke", "planted", 99, n_instruments=20, n_bars=1000, ic=0.15)
    R, _ = e3.returns_matrix(st, cfg, trials, zero_costs())
    sel, sr, sr0, d = _select(R)
    assert trials[sel]["feature"] == "signal_x_ewma"
    assert d > 0.95
