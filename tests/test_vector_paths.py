"""The vectorized engine's contiguous fast path equals its general path bit for bit, and the
general path (stores with gaps) agrees with the event engine."""

import numpy as np
from engine_helpers import CAL, run_both

from quant_research_engine.costs import CostConfig
from quant_research_engine.data.csvio import make_table
from quant_research_engine.store.builder import build_store
from quant_research_engine.validation.scenarios import make_scenario
from quant_research_engine.vector import _contiguous, run_target_shares


def test_fast_path_equals_general_path():
    for s in range(1, 41):
        sc = make_scenario(s, 2)
        p = sc.store.panel()
        assert _contiguous(p)
        kw = dict(fill_delay_bars=sc.fill_delay, initial_cash=1e6)
        a = run_target_shares(p, sc.dec_idx, sc.targets, **kw)
        b = run_target_shares(p, sc.dec_idx, sc.targets, force_general=True, **kw)
        assert np.array_equal(a.positions, b.positions) and np.array_equal(a.equity, b.equity)
        for d1, d2 in ((a.fills, b.fills), (a.orders, b.orders)):
            for k in d1:
                assert np.array_equal(d1[k], d2[k]), (s, k)


def test_general_path_with_a_gap_matches_the_event_engine():
    rows = {
        k: []
        for k in (
            "instrument_id",
            "ts_open",
            "ts_event",
            "ts_avail",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
    }
    for iid, skip in (("X", None), ("Y", 3)):
        for k in range(8):
            if k == skip:
                continue
            c = 10.0 + k + (0.5 if iid == "Y" else 0.0)
            for key, v in (
                ("instrument_id", iid),
                ("ts_open", int(CAL.ts_open[k])),
                ("ts_event", int(CAL.ts_event[k])),
                ("ts_avail", int(CAL.ts_event[k])),
                ("open", c - 0.2),
                ("high", c + 0.1),
                ("low", c - 0.3),
                ("close", c),
                ("volume", 1e6),
            ):
                rows[key].append(v)
    ins = make_table(
        "instruments",
        instrument_id=["X", "Y"],
        symbol=["X", "Y"],
        asset_class=["equity"] * 2,
        currency=["USD"] * 2,
        ts_list=[int(CAL.ts_event[0])] * 2,
        ts_delist=[None, None],
        delist_return=[None, None],
        lot_size=[0.0, 0.0],
        tick_size=[0.01] * 2,
        sector=["", ""],
    )
    st = build_store(
        ins,
        make_table("bars", **rows),
        make_table(
            "corporate_actions", instrument_id=[], action=[], ts_ex=[], ts_avail=[], value=[]
        ),
        make_table("series", series_id=[], ts_event=[], ts_avail=[], vintage=[], value=[]),
        {"calendar": "equity_daily", "ppy": 252},
    )
    assert not _contiguous(st.panel())
    sched = {0: {"X": 100.0, "Y": -50.0}, 2: {"X": 40.0, "Y": 80.0}, 4: {"X": 0.0, "Y": 10.0}}
    for delay in (1, 2):
        ev, vr = run_both(st, sched, costs=CostConfig(impact={"model": "none"}), delay=delay)
        assert np.allclose(ev.equity, vr.equity, rtol=1e-12, atol=0)
        assert len(ev.fills) == len(vr.fills["quantity"])
