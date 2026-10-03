"""Check 2: as-of semantics (exact)."""

import numpy as np
from helpers import T, golden_store, random_series_rows

from quant_research_engine.data import NO_TS, make_table
from quant_research_engine.rng import stream
from quant_research_engine.store import Store, asof_vintage
from quant_research_engine.store.asof import asof_vintage_bruteforce
from quant_research_engine.store.duck import asof_series_duckdb, connect

STEP = 3_600_000_000


def test_asof_vintage_equals_bruteforce_on_500_random_series():
    rng = stream(11, "test.asof")
    checked = 0
    for _ in range(170):
        tab = random_series_rows(rng, n_series=3, n_periods=4)
        cols = tab.columns
        for t in [-1, 0, *rng.integers(0, 40, size=6).tolist(), 1000]:
            ti = int(t) * STEP
            a = asof_vintage(cols, ti)
            b = asof_vintage_bruteforce(cols, ti)
            assert a.tolist() == b.tolist()
        checked += len(set(cols["series_id"].tolist()))
    assert checked >= 500


def test_asof_is_inclusive_at_ts_avail():
    tab = random_series_rows(stream(3, "test.asof.incl"))
    cols = tab.columns
    for i in range(len(tab)):
        sel = asof_vintage(cols, int(cols["ts_avail"][i]))
        # the row released exactly at t is known at t
        assert i in sel.tolist() or any(
            cols["series_id"][j] == cols["series_id"][i]
            and cols["ts_event"][j] == cols["ts_event"][i]
            and cols["vintage"][j] > cols["vintage"][i]
            for j in sel.tolist()
        )


def test_duckdb_asof_join_equals_asof_vintage():
    rng = stream(12, "test.asof.duck")
    for _ in range(25):
        tab = random_series_rows(rng, n_series=4, n_periods=5)
        store = Store({"series": tab}, {"calendar": "equity_daily", "ppy": 252})
        con = connect(store)
        try:
            for t in rng.integers(-1, 45, size=8).tolist():
                ti = int(t) * STEP
                ref = asof_vintage(tab.columns, ti)
                got = asof_series_duckdb(con, ti)
                assert got["series_id"].tolist() == tab["series_id"][ref].tolist()
                assert got["ts_event"].tolist() == tab["ts_event"][ref].tolist()
                assert got["vintage"].tolist() == tab["vintage"][ref].tolist()
                assert got["value"].tolist() == tab["value"][ref].tolist()
        finally:
            con.close()


def _random_instruments(rng, n):
    ids = [f"I{k:02d}" for k in range(n)]
    lst = rng.integers(0, 50, size=n) * STEP
    dl = []
    dr = []
    for k in range(n):
        if rng.random() < 0.4:
            dl.append(int(lst[k] + rng.integers(0, 30) * STEP))
            dr.append(float(-rng.random()))
        else:
            dl.append(None)
            dr.append(None)
    return make_table(
        "instruments",
        instrument_id=ids,
        symbol=ids,
        asset_class=["equity"] * n,
        currency=["USD"] * n,
        ts_list=lst.tolist(),
        ts_delist=dl,
        delist_return=dr,
        lot_size=[0.0] * n,
        tick_size=[0.01] * n,
        sector=[""] * n,
    )


def test_instruments_at_equals_bruteforce():
    rng = stream(13, "test.universe")
    for _ in range(100):
        ins = _random_instruments(rng, int(rng.integers(1, 15)))
        store = Store({"instruments": ins}, {})
        for t in rng.integers(-2, 90, size=10).tolist():
            ti = int(t) * STEP
            brute = [
                ins["instrument_id"][k]
                for k in range(len(ins))
                if ins["ts_list"][k] <= ti
                and (ins["ts_delist"][k] == NO_TS or ins["ts_delist"][k] > ti)
            ]
            assert store.instruments_at(ti) == sorted(brute)
            known = store.instruments(knowledge=ti)
            for k in range(len(known)):
                d = known["ts_delist"][k]
                assert d == NO_TS or d <= ti  # delisting fields hidden until ts_delist
                assert np.isnan(known["delist_return"][k]) == (d == NO_TS)


def test_appendix_a_eps_revision():
    s = golden_store()
    a = s.series("A.eps", knowledge=T("2024-03-04T21:00:00Z"))
    assert a["value"].tolist() == [1.1] and a["vintage"].tolist() == [0]
    b = s.series("A.eps", knowledge=T("2024-03-05T21:00:00Z"))
    assert b["value"].tolist() == [1.05]
    c = s.series("A.eps", knowledge=T("2024-02-15T20:59:59Z"))
    assert c["value"].tolist() == []
    d = s.series("A.eps", knowledge=T("2024-02-15T21:00:00Z"))  # inclusive
    assert d["value"].tolist() == [1.1]


def test_queries_never_return_rows_after_knowledge():
    s = golden_store()
    for t in ("2024-03-04T21:00:00Z", "2024-03-05T20:00:00Z", "2024-03-05T21:00:00Z"):
        k = T(t)
        for tab in s.bars(["A", "B", "ZZ"], 10, knowledge=k).values():
            assert (tab["ts_avail"] <= k).all()
        assert (s.corporate_actions(knowledge=k)["ts_avail"] <= k).all()
        assert (s.liquidity(knowledge=k)["ts_avail"] <= k).all()
    assert len(s.bars(["A"], 0, knowledge=T("2024-03-06T21:00:00Z"))["A"]) == 0
    assert len(s.bars(["A"], -5, knowledge=T("2024-03-06T21:00:00Z"))["A"]) == 0
    assert len(s.bars(["A"], 10**6, knowledge=T("2024-03-06T21:00:00Z"))["A"]) == 3
