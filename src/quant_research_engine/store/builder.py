"""Assemble a store from its tables; the liquidity dataset is derived by the pure function."""

from __future__ import annotations

import numpy as np

from quant_research_engine.data.csvio import Table, make_table, sort_table, validate_table
from quant_research_engine.data.schemas import SCHEMAS
from quant_research_engine.store.adjust import bar_events, bar_returns
from quant_research_engine.store.liquidity import liquidity_rows
from quant_research_engine.store.store import Store


def liquidity_table(bars: Table, actions: Table) -> Table:
    """The ``liquidity`` dataset of every bar (rows only where the function defines one)."""
    acts: dict[str, list[tuple[str, int, float]]] = {}
    for i in range(len(actions)):
        acts.setdefault(str(actions["instrument_id"][i]), []).append(
            (str(actions["action"][i]), int(actions["ts_ex"][i]), float(actions["value"][i]))
        )
    iid = bars["instrument_id"]
    rows = {k: [] for k in ("ts_event", "ts_avail", "instrument_id", "adv_shares", "sigma_bar")}
    if len(bars):
        starts = np.flatnonzero(np.r_[True, iid[1:] != iid[:-1]])
        ends = np.r_[starts[1:], len(bars)]
        for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
            name = str(iid[a])
            ratio, div = bar_events(bars["ts_open"][a:b], acts.get(name, []))
            rets = bar_returns(bars["close"][a:b], ratio, div)
            valid, adv, sig = liquidity_rows(bars["volume"][a:b], rets, ratio)
            for j in np.flatnonzero(valid).tolist():
                rows["ts_event"].append(int(bars["ts_event"][a + j]))
                rows["ts_avail"].append(int(bars["ts_avail"][a + j]))
                rows["instrument_id"].append(name)
                rows["adv_shares"].append(float(adv[j]))
                rows["sigma_bar"].append(float(sig[j]))
    return sort_table(make_table(SCHEMAS["liquidity"], **rows))


def build_store(
    instruments: Table,
    bars: Table,
    corporate_actions: Table,
    series: Table,
    meta: dict,
) -> Store:
    for t in (instruments, bars, corporate_actions, series):
        validate_table(t)
    liq = liquidity_table(bars, corporate_actions)
    tables = {
        "instruments": instruments,
        "bars": bars,
        "corporate_actions": corporate_actions,
        "series": series,
        "liquidity": liq,
    }
    return Store(tables, dict(meta))
