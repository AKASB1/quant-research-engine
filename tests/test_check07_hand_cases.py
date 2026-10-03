"""Check 7 hand-computed cases and check 3 (engine part), on both engines."""

import os

import numpy as np
from conftest import GOLDEN
from engine_helpers import run_both, tiny_store
from helpers import golden_store

from quant_research_engine.costs import CostConfig
from quant_research_engine.data import load_csv

DAY = 1.0 / 365.0


def both_equal(ev, vr):
    assert np.allclose(ev.equity, vr.equity, rtol=1e-12, atol=0)
    return ev.equity


def test_one_buy_reproduces_appendix_a():
    st = golden_store()
    p = st.panel()
    p.adv[0, :] = [1e6, 250000]
    p.sigma[0, :] = [0.02, 0.015]
    p.liq_avail[0, :] = p.ts_event[0]
    ev, vr = run_both(st, {0: {"A": 100.0}}, costs=CostConfig(), cash=100000.0)
    gold = load_csv(os.path.join(GOLDEN, "equity_v1.csv"))
    for col in gold.schema.names[1:]:
        assert np.allclose(
            [r[gold.schema.names.index(col)] for r in ev.equity_rows],
            gold[col],
            rtol=1e-12,
            atol=1e-12,
        ), col
    assert np.allclose(vr.cash, gold["cash"], rtol=1e-12)
    f = ev.fills[0]
    assert f[4:] == (100.0, 101.5, 101.53045, 2.03, 1.015, 1.015)


def test_round_trip():
    st = tiny_store(
        {"X": [(10.0, 10.0, 1e6), (11.0, 12.0, 1e6), (13.0, 14.0, 1e6), (15.0, 15.0, 1e6)]}
    )
    ev, vr = run_both(st, {0: {"X": 100.0}, 1: {"X": 0.0}})
    eq = both_equal(ev, vr)
    # buy at 11 (open of bar 1), sell at 13 (open of bar 2): +200
    assert eq.tolist() == [100000.0, 100100.0, 100200.0, 100200.0]


def test_short_with_borrow():
    st = tiny_store(
        {"X": [(10.0, 10.0, 1e6), (10.0, 10.0, 1e6), (10.0, 10.0, 1e6), (10.0, 10.0, 1e6)]}
    )
    costs = CostConfig(
        commission_bps=0, half_spread_bps=0, impact={"model": "none"}, financing_bps_annual=0
    )
    ev, vr = run_both(st, {0: {"X": -100.0}}, costs=costs)
    both_equal(ev, vr)
    rows = ev.equity_rows
    # bar 2 accrues borrow on 100 shares at 10 for one day at 50 bp per year
    assert abs(rows[2][9] - 100 * 10 * 50 / 1e4 * DAY) < 1e-12
    assert rows[1][9] == 0.0


def test_split_in_the_middle_of_a_holding_leaves_equity_unchanged():
    # 2-for-1 split effective in bar 2: quotes halve, holdings double
    st = tiny_store(
        {"X": [(10.0, 10.0, 1e6), (10.0, 10.0, 1e6), (5.0, 5.0, 1e6), (5.0, 5.0, 1e6)]},
        actions=[("X", "split", 2, 2.0, 0)],
    )
    ev, vr = run_both(st, {0: {"X": 100.0}})
    eq = both_equal(ev, vr)
    assert eq.tolist() == [100000.0] * 4
    assert ev.positions_rows[-1][2] == 200.0
    assert vr.positions[-1, 0] == 200.0


def test_pending_order_rescaled_across_a_split():
    # decision at bar 0, fill delay 2 -> fill in bar 2, which is the split's ex-bar
    st = tiny_store(
        {"X": [(10.0, 10.0, 1e6), (10.0, 10.0, 1e6), (5.0, 5.0, 1e6), (5.0, 5.0, 1e6)]},
        actions=[("X", "split", 2, 2.0, 0)],
    )
    ev, vr = run_both(st, {0: {"X": 100.0}}, delay=2)
    both_equal(ev, vr)
    assert ev.fills[0][4] == 200.0 and ev.fills[0][5] == 5.0
    assert vr.fills["quantity"].tolist() == [200.0]


def test_delisting_with_negative_return():
    st = tiny_store(
        {"X": [(10.0, 10.0, 1e6), (10.0, 10.0, 1e6), (10.0, 8.0, 1e6)]}, delist={"X": -0.5}
    )
    ev, vr = run_both(st, {0: {"X": 100.0}})
    eq = both_equal(ev, vr)
    # exit at 8 * 0.5 = 4: equity falls by 100 * (10 - 4)
    assert eq[-1] == 100000.0 - 600.0
    assert ev.fills[-1][1] == "DELIST" and ev.fills[-1][4] == -100.0 and ev.fills[-1][5] == 4.0
    assert not ev.positions_rows or ev.positions_rows[-1][0] != int(st.panel().ts_event[2])


def test_dividend_paid_to_longs_and_charged_to_shorts():
    prices = [(10.0, 10.0, 1e6), (10.0, 10.0, 1e6), (9.5, 9.5, 1e6), (9.5, 9.5, 1e6)]
    st = tiny_store(
        {"L": prices, "S": prices},
        actions=[("L", "cash_dividend", 2, 0.5, 0), ("S", "cash_dividend", 2, 0.5, 0)],
    )
    costs = CostConfig(
        commission_bps=0,
        half_spread_bps=0,
        impact={"model": "none"},
        borrow_bps_annual=0,
        financing_bps_annual=0,
    )
    ev, vr = run_both(st, {0: {"L": 100.0, "S": -100.0}}, costs=costs)
    both_equal(ev, vr)
    assert ev.equity_rows[2][11] == 0.0  # +50 long, -50 short
    assert ev.per_instrument["L"]["dividends"] == 50.0
    assert ev.per_instrument["S"]["dividends"] == -50.0
    # the ex-day price drop and the dividend cancel for the long
    assert ev.per_instrument["L"]["hold_pnl"] == -50.0
