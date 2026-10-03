"""Random engine scenarios for the equivalence check and E1 (stream ``e1.scenario``).

A scenario is a small store (2 to 12 instruments, 50 to 300 bars) with splits (ratios 2 and 3,
some between a decision and its fill), cash dividends, delistings with negative returns, late
listings, zero-volume bars, and lot sizes, plus a target-share schedule with sign changes,
``fill_delay_bars`` of 1 to 3, and decisions every 1 to 5 bars. Targets are rounded to lots and
to the quantity grid, are zero outside an instrument's listed life, and are non-negative when
shorts are off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from quant_research_engine.data.csvio import NO_TS, make_table
from quant_research_engine.engine.quantity import round_lot
from quant_research_engine.rng import stream
from quant_research_engine.store.builder import build_store
from quant_research_engine.store.calendars import equity_daily


@dataclass
class Scenario:
    seed: int
    index: int
    store: object
    dec_idx: np.ndarray
    targets: np.ndarray
    fill_delay: int
    allow_short: bool
    rebalance_every: int
    features: dict


def make_scenario(seed: int, index: int = 0) -> Scenario:
    rng = stream(seed * 1000 + index, "e1.scenario")
    n = int(rng.integers(2, 13))
    T = int(rng.integers(50, 301))
    cal = equity_daily(date(2020, 1, 2), T)
    ids = [f"S{i:02d}" for i in range(n)]
    lots = rng.choice([0.0, 1.0, 10.0, 100.0], size=n)
    first = np.where(rng.random(n) < 0.3, rng.integers(1, max(2, T // 2), n), 0)
    last = np.full(n, T - 1)
    delist = rng.random(n) < 0.3
    for i in range(n):
        if delist[i] and first[i] + 10 < T - 1:
            last[i] = int(rng.integers(first[i] + 10, T))
        else:
            delist[i] = False
    dret = np.where(delist, rng.uniform(-0.95, 0.1, n), np.nan)
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
    acts = {k: [] for k in ("instrument_id", "action", "ts_ex", "ts_avail", "value")}
    feats = {
        "splits": 0,
        "dividends": 0,
        "delistings": int(delist.sum()),
        "late_listings": int((first > 0).sum()),
        "zero_volume_bars": 0,
        "lots": int((lots > 0).sum()),
    }
    for i in range(n):
        ks = np.arange(first[i], last[i] + 1)
        m = ks.size
        split_bars = set()
        for _ in range(int(rng.integers(0, 3))):
            if m > 8:
                split_bars.add(int(rng.integers(first[i] + 3, last[i] + 1)))
        div_bars = set()
        for _ in range(int(rng.integers(0, 4))):
            if m > 8:
                div_bars.add(int(rng.integers(first[i] + 3, last[i] + 1)))
        c = float(rng.uniform(5, 200))
        ratio_at = {}
        div_at = {}
        for k in sorted(split_bars):
            ratio_at[k] = float(rng.choice([2.0, 3.0]))
        for k in sorted(div_bars):
            div_at[k] = round(float(rng.uniform(0.01, 0.02)) * c, 4)
        for k in ks.tolist():
            r = rng.normal(0.0, 0.02)
            ron = rng.normal(0.0, 0.01)
            d = div_at.get(k, 0.0)
            ratio = ratio_at.get(k, 1.0)
            o = ((1 + ron) * c - d) / ratio
            cl = ((1 + r) * c - d) / ratio
            if o <= 0 or cl <= 0:
                o, cl = c / ratio, c / ratio
            hi = max(o, cl) * (1 + abs(rng.normal(0, 0.005)))
            lo = min(o, cl) * (1 - abs(rng.normal(0, 0.005)))
            vol = (
                0.0
                if (k > first[i] and rng.random() < 0.05)
                else float(round(math.exp(rng.normal(14, 1))))
            )
            feats["zero_volume_bars"] += vol == 0.0
            for key, val in (
                ("instrument_id", ids[i]),
                ("ts_open", int(cal.ts_open[k])),
                ("ts_event", int(cal.ts_event[k])),
                ("ts_avail", int(cal.ts_event[k])),
                ("open", o),
                ("high", hi),
                ("low", lo),
                ("close", cl),
                ("volume", vol),
            ):
                rows[key].append(val)
            c = cl
            if k in ratio_at:
                ann = max(int(first[i]), k - int(rng.integers(1, 6)))
                ann_ts = int(cal.ts_event[ann]) if ann < k else int(cal.ts_open[k])
                for key, val in (
                    ("instrument_id", ids[i]),
                    ("action", "split"),
                    ("ts_ex", int(cal.ts_open[k])),
                    ("ts_avail", ann_ts),
                    ("value", ratio_at[k]),
                ):
                    acts[key].append(val)
                feats["splits"] += 1
            if k in div_at and div_at[k] > 0:
                ann = max(int(first[i]), k - int(rng.integers(1, 6)))
                ann_ts = int(cal.ts_event[ann]) if ann < k else int(cal.ts_open[k])
                for key, val in (
                    ("instrument_id", ids[i]),
                    ("action", "cash_dividend"),
                    ("ts_ex", int(cal.ts_open[k])),
                    ("ts_avail", ann_ts),
                    ("value", div_at[k]),
                ):
                    acts[key].append(val)
                feats["dividends"] += 1
    ins = make_table(
        "instruments",
        instrument_id=ids,
        symbol=ids,
        asset_class=["equity"] * n,
        currency=["USD"] * n,
        ts_list=[int(cal.ts_event[f]) for f in first],
        ts_delist=[int(cal.ts_event[last[i]]) if delist[i] else None for i in range(n)],
        delist_return=[float(dret[i]) if delist[i] else None for i in range(n)],
        lot_size=lots.tolist(),
        tick_size=[0.01] * n,
        sector=[""] * n,
    )
    from quant_research_engine.data.csvio import sort_table

    bars = make_table("bars", **rows)
    ca = sort_table(make_table("corporate_actions", **acts))
    ser = make_table("series", series_id=[], ts_event=[], ts_avail=[], vintage=[], value=[])
    store = build_store(
        ins,
        bars,
        ca,
        ser,
        {
            "calendar": "equity_daily",
            "ppy": 252,
            "generator": {"name": "e1.scenario"},
            "seed": seed,
        },
    )
    p = store.panel()
    delay = int(rng.integers(1, 4))
    every = int(rng.integers(1, 6))
    allow_short = bool(rng.random() < 0.6)
    start = int(rng.integers(0, 4))
    dec = np.arange(start, p.shape[0], every)
    tg = np.zeros((dec.size, n))
    scale = rng.uniform(50, 3000, n)
    for j, k in enumerate(dec.tolist()):
        t = int(p.ts_event[k])
        for i in range(n):
            listed = p.ts_list[i] <= t and (p.ts_delist[i] == NO_TS or p.ts_delist[i] > t)
            has_bar = p.first_bar[i] >= 0 and p.first_bar[i] <= k
            if not (listed and has_bar):
                continue
            u = rng.random()
            if u < 0.15:
                q = 0.0
            elif u < 0.3 and j > 0:
                q = tg[j - 1, i]  # unchanged target (no order unless a split moved it)
            else:
                q = rng.normal(0, 1) * scale[i]
            if not allow_short:
                q = abs(q)
            tg[j, i] = float(round_lot(q, lots[i]))
    return Scenario(seed, index, store, dec, tg, delay, allow_short, every, feats)
