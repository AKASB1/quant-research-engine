"""Tiny hand-made stores for engine tests (equity_daily sessions from 2024-03-04)."""

from __future__ import annotations

from datetime import date

import numpy as np

from quant_research_engine.context import TargetShares
from quant_research_engine.costs import CostConfig, zero_costs
from quant_research_engine.data.csvio import make_table, sort_table
from quant_research_engine.engine import EngineConfig, EventEngine
from quant_research_engine.store.builder import build_store
from quant_research_engine.store.calendars import equity_daily
from quant_research_engine.vector import run_target_shares

CAL = equity_daily(date(2024, 3, 4), 40)


def tiny_store(
    prices: dict[str, list[tuple]],
    actions=(),
    delist: dict | None = None,
    lots=None,
    first: dict | None = None,
    liquidity=(1e6, 0.02),
):
    """``prices[id]`` = list of (open, close, volume) per bar from bar ``first[id]`` (default 0);
    ``actions`` = (id, action, bar, value, announce_bar); ``delist[id]`` = delist_return (the
    instrument delists at its last bar)."""
    delist = delist or {}
    first = first or {}
    lots = lots or {}
    ids = sorted(prices)
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
    for iid in ids:
        f = first.get(iid, 0)
        for j, (o, c, v) in enumerate(prices[iid]):
            k = f + j
            rows["instrument_id"].append(iid)
            rows["ts_open"].append(int(CAL.ts_open[k]))
            rows["ts_event"].append(int(CAL.ts_event[k]))
            rows["ts_avail"].append(int(CAL.ts_event[k]))
            rows["open"].append(o)
            rows["high"].append(max(o, c))
            rows["low"].append(min(o, c))
            rows["close"].append(c)
            rows["volume"].append(v)
    ins = make_table(
        "instruments",
        instrument_id=ids,
        symbol=ids,
        asset_class=["equity"] * len(ids),
        currency=["USD"] * len(ids),
        ts_list=[int(CAL.ts_event[first.get(i, 0)]) for i in ids],
        ts_delist=[
            int(CAL.ts_event[first.get(i, 0) + len(prices[i]) - 1]) if i in delist else None
            for i in ids
        ],
        delist_return=[delist.get(i) for i in ids],
        lot_size=[float(lots.get(i, 0.0)) for i in ids],
        tick_size=[0.01] * len(ids),
        sector=[""] * len(ids),
    )
    ca = {k: [] for k in ("instrument_id", "action", "ts_ex", "ts_avail", "value")}
    for iid, action, bar, value, ann in actions:
        ca["instrument_id"].append(iid)
        ca["action"].append(action)
        ca["ts_ex"].append(int(CAL.ts_open[bar]))
        ca["ts_avail"].append(int(CAL.ts_event[ann]))
        ca["value"].append(float(value))
    st = build_store(
        ins,
        make_table("bars", **rows),
        sort_table(make_table("corporate_actions", **ca)),
        make_table("series", series_id=[], ts_event=[], ts_avail=[], vintage=[], value=[]),
        {"calendar": "equity_daily", "ppy": 252},
    )
    p = st.panel()
    if liquidity is not None:  # given liquidity inputs (not derived), known at every bar
        p.adv[:] = np.where(p.has_bar, liquidity[0], np.nan)
        p.sigma[:] = np.where(p.has_bar, liquidity[1], np.nan)
        p.liq_avail[:] = np.where(p.has_bar, p.ts_event[:, None], np.iinfo(np.int64).max)
    return st


def run_both(
    st,
    schedule: dict[int, dict[str, float]],
    costs: CostConfig | None = None,
    cash=100_000.0,
    delay=1,
    fill_model="next_open",
    allow_short=True,
):
    costs = costs if costs is not None else zero_costs()
    cfg = EngineConfig(
        fill_model=fill_model,
        fill_delay_bars=delay,
        initial_cash=cash,
        costs=costs,
        allow_short=allow_short,
    )
    p = st.panel()
    eng = EventEngine(p, cfg)
    for k in range(p.shape[0]):
        eng.process_bar(k)
        if k in schedule:
            eng.submit(TargetShares(schedule[k]), st.instruments_at(int(p.ts_event[k])))
    ev = eng.result()
    dec = sorted(schedule)
    tg = np.zeros((len(dec), len(p.ids)))
    for j, k in enumerate(dec):
        for iid, q in schedule[k].items():
            tg[j, p.ids.index(iid)] = q
    vr = run_target_shares(
        p, dec, tg, fill_model=fill_model, fill_delay_bars=delay, initial_cash=cash, costs=costs
    )
    return ev, vr
