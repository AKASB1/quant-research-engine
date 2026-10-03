"""Check 8: the cost model (hand cases, monotonicity, no look-ahead in impact)."""

import math

import numpy as np
from engine_helpers import run_both, tiny_store

from quant_research_engine.context import Orders, OrderSpec
from quant_research_engine.costs import CostConfig, ImpactCfg, fill_costs
from quant_research_engine.engine import EngineConfig, EventEngine


def one(q, m, sigma, V, cfg):
    return tuple(float(x) for x in fill_costs(q, m, sigma, V, cfg))


def test_appendix_a_known_answer():
    price, sp, im, cm, un = one(100, 101.5, 0.02, 1e6, CostConfig())
    assert abs(price - 101.53045) <= 1e-9 * 101.53045
    assert abs(sp - 2.03) <= 1e-9 and abs(im - 1.015) <= 1e-9 and abs(cm - 1.015) <= 1e-9
    assert un == 0.0


def test_short_sale():
    price, sp, im, cm, _ = one(-100, 101.5, 0.02, 1e6, CostConfig())
    assert abs(price - 101.5 * (1 - 3e-4)) <= 1e-9 * 101.5
    assert (sp, im, cm) == one(100, 101.5, 0.02, 1e6, CostConfig())[1:4]


def test_min_commission_and_per_share():
    cfg = CostConfig(min_commission=5.0, commission_per_share=0.01)
    assert one(10, 20.0, 0.02, 1e6, cfg)[3] == 5.0  # 0.02 + 0.1 < 5
    big = one(10_000, 20.0, 0.02, 1e6, cfg)[3]
    assert abs(big - (10_000 * 20.0 * 1e-4 + 10_000 * 0.01)) <= 1e-9


def test_linear_impact_and_none():
    lin = CostConfig(impact=ImpactCfg(model="linear", y=0.5))
    assert (
        abs(one(1000, 10.0, 0.02, 1e6, lin)[2] - 1000 * 10 * (1e4 * 0.5 * 0.02 * 1000 / 1e6) / 1e4)
        < 1e-12
    )
    assert one(1000, 10.0, 0.02, 1e6, CostConfig(impact=ImpactCfg(model="none")))[2] == 0.0


def test_unknown_liquidity_gives_zero_impact_and_is_counted():
    _, _, im, _, un = one(100, 10.0, math.nan, 1e6, CostConfig())
    assert im == 0.0 and un == 1.0
    _, _, im, _, un = one(100, 10.0, 0.02, math.nan, CostConfig())
    assert im == 0.0 and un == 1.0


def test_lot_rounding_of_targets():
    st = tiny_store({"X": [(10.0, 10.0, 1e6)] * 4}, lots={"X": 100.0})
    ev, vr = run_both(st, {0: {"X": 0.0}})
    eng = EventEngine(st.panel(), EngineConfig(initial_cash=1e5))
    eng.process_bar(0)
    from quant_research_engine.context import TargetWeights

    eng.submit(TargetWeights({"X": 0.5}), ["X"])  # 0.5 * 1e5 / 10 = 5000 -> lots of 100
    assert eng.pending[0][0].qty == 5000.0
    eng.submit(TargetWeights({"X": 0.5037}), ["X"])  # 5037 -> 5000: no new order
    assert len(eng.pending[0]) == 1
    del ev, vr


def test_participation_cap_binds_with_gtc_carry_and_day_cancel():
    st = tiny_store(
        {"X": [(10.0, 10.0, 1e6), (10.0, 10.0, 1000.0), (10.0, 10.0, 0.0), (10.0, 10.0, 1000.0)]}
    )
    for tif, expect in (("gtc", [100.0, 100.0]), ("day", [100.0])):
        eng = EventEngine(st.panel(), EngineConfig(initial_cash=1e6))
        eng.process_bar(0)
        eng.submit(Orders((OrderSpec("X", 200.0, tif),)), ["X"])
        for k in range(1, 4):
            eng.process_bar(k)
        r = eng.result()
        assert [f[4] for f in r.fills] == expect, tif
        assert r.cap_binds >= 1
        assert eng.E[0] == sum(expect)


def test_cost_monotone_in_quantity_sigma_and_liquidity():
    cfg = CostConfig()
    qs = np.array([1, 10, 100, 1000, 10_000], dtype=float)
    tot = lambda q, s, v: sum(one(q, 50.0, s, v, cfg)[1:4])  # noqa: E731
    assert all(tot(a, 0.02, 1e6) <= tot(b, 0.02, 1e6) for a, b in zip(qs[:-1], qs[1:], strict=True))
    assert all(tot(100, a, 1e6) <= tot(100, b, 1e6) for a, b in ((0.01, 0.02), (0.02, 0.05)))
    assert all(tot(100, 0.02, a) >= tot(100, 0.02, b) for a, b in ((1e5, 1e6), (1e6, 1e7)))


def test_impact_ignores_fill_bar_volume_and_future_data():
    base = [(10.0, 10.0, 1e6), (10.0, 10.0, 1e6), (10.0, 10.0, 1e6)]
    other = [(10.0, 10.0, 1e6), (10.0, 10.0, 5e6), (10.0, 10.0, 7e5)]
    a, _ = run_both(tiny_store({"X": base}), {0: {"X": 100.0}}, costs=CostConfig())
    b, _ = run_both(tiny_store({"X": other}), {0: {"X": 100.0}}, costs=CostConfig())
    assert a.fills[0][7:10] == b.fills[0][7:10]


def test_decision_time_liquidity_is_unchanged_by_the_poison():
    """Impact uses the liquidity known at the decision: identical in the real and the poisoned
    world for every decision up to the poisoning instant."""
    import os

    from quant_research_engine.engine import EventEngine as EE
    from quant_research_engine.guards.poison import poison_store
    from quant_research_engine.synth import generate, load_synth_config

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    st, _ = generate(load_synth_config(os.path.join(root, "configs", "synth", "small.json")), 2)
    p = st.panel()
    real = EE(p, EngineConfig())
    for k in (60, 200, 400):
        w = poison_store(st, int(p.ts_event[k]), 99 + k)
        pw = w.panel()
        e2 = EE(pw, EngineConfig())
        for iid in st.instruments_at(int(p.ts_event[k])):
            i, j = p.ids.index(iid), pw.ids.index(iid)
            assert np.array_equal(real.liq_adv[: k + 1, i], e2.liq_adv[: k + 1, j], equal_nan=True)
            assert np.array_equal(
                real.liq_sigma[: k + 1, i], e2.liq_sigma[: k + 1, j], equal_nan=True
            )
