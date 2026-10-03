"""Tier 2 items 3 and 4: capacity and impact, effective number of trials and multiple testing."""

import math

import numpy as np

from quant_research_engine.experiments import t2_capacity, t2_trials


def test_holm_and_bh_known_answers():
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.2])
    # Holm at 0.05: sorted 0.005 <= 0.01, 0.01 <= 0.0125, 0.03 > 0.0167 -> stop: two rejections
    assert t2_trials.holm(p, 0.05).tolist() == [True, False, False, True, False]
    # BH at 0.05: thresholds 0.01, 0.02, 0.03, 0.04, 0.05; largest k with p_(k) <= k q/m is 4
    assert t2_trials.bh(p, 0.05).tolist() == [True, True, True, True, False]
    assert not t2_trials.holm(np.array([0.5, 0.9]), 0.05).any()
    assert not t2_trials.bh(np.array([0.5, 0.9]), 0.05).any()


def test_effective_number_of_trials():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((5000, 1))
    same = np.hstack([z, 2 * z, z + 1e-12 * rng.standard_normal((5000, 1))])
    assert abs(t2_trials.n_eff(same) - 1.0) < 1e-6
    indep = rng.standard_normal((5000, 10))
    assert 9.5 < t2_trials.n_eff(indep) <= 10.0
    # two independent blocks of five identical trials: two effective trials
    a, b = rng.standard_normal((5000, 1)), rng.standard_normal((5000, 1))
    blocks = np.hstack([a] * 5 + [b] * 5)
    assert abs(t2_trials.n_eff(blocks) - 2.0) < 0.01


def test_capacity_crossing_interpolates_in_log_capital():
    caps = [1e6, 1e7, 1e8]
    assert math.isclose(t2_capacity._crossing(caps, [1.0, 0.5, -0.5], 0.0), 10**7.5)
    assert math.isclose(t2_capacity._crossing(caps, [1.0, 0.5, -0.5], 0.5), 1e7)
    assert t2_capacity._crossing(caps, [1.0, 0.9, 0.8], 0.0) is None


def test_capacity_job_on_a_small_market():
    cfg = {"capitals": [1e5, 1e10], "n_bars": 300, "n_instruments": 12, "rebalance": 5,
           "warmup": 40, "seeds": [1, 1]}  # fmt: skip
    rows = t2_capacity.run_job(("seed", 1), cfg, "quick", None)
    assert [r["capital"] for r in rows] == [1e5, 1e10]
    small, large = rows
    assert 0 <= small["cap_bound_share"] < large["cap_bound_share"] <= 1
    # impact relative to equity need not grow with capital: once the cap binds, trading shrinks
    assert small["impact_drag_annual"] > 0 and large["impact_drag_annual"] > 0
    assert small["cost_drag_annual"] > 0 and large["cost_drag_annual"] > 0
    for r in rows:
        assert math.isfinite(r["net_sharpe"]) and math.isfinite(r["gross_sharpe"])
        assert r["max_identity_residual"] < 1e-9
