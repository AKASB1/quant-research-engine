"""Poisoned copies of a store for the replay audit (guard G2).

``poison_store(store, t, seed)`` leaves every row on its side of ``t``: a row known at ``t``
stays known with the same value, and a row not yet known stays unknown. What lies after ``t``
changes (stream ``audit.poison``):

- bars with ``ts_event > t``: prices of each bar multiplied by one factor, log-uniform in
  [0.1, 10] (high and low still bound open and close), volumes redrawn (some zero);
- future delistings: cancelled for some instruments (bars continue to the end of the calendar),
  added for others (``ts_delist`` = the ``ts_event`` of the new last bar, at least one bar after
  ``t``), so the set of final survivors differs; instruments listing after ``t`` may be removed,
  and new ones are added;
- series rows with ``ts_avail > t`` get redrawn values; for every period with a vintage known at
  ``t``, later vintages are changed, removed, or appended (vintages stay consecutive), so the
  last vintage of a known period differs;
- corporate actions announced after ``t`` are changed, removed, or added (one announced at or
  before ``t`` stays);
- liquidity rows are recomputed from the poisoned bars (rows known at ``t`` are unchanged; the
  builder checks this), and the manifest names another generator seed and other parameters.

The calendar, ``ppy``, and the schema fields are the same in both worlds.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from quant_research_engine.data.csvio import NO_TS, Table, sort_table, validate_table
from quant_research_engine.data.schemas import SCHEMAS
from quant_research_engine.rng import stream
from quant_research_engine.store.builder import liquidity_table
from quant_research_engine.store.store import Store


def _rows(table: Table) -> list[dict]:
    names = table.schema.names
    cols = [table[n].tolist() for n in names]
    return [dict(zip(names, r, strict=True)) for r in zip(*cols, strict=True)]


def _table(schema: str, rows: list[dict]) -> Table:
    sch = SCHEMAS[schema]
    cols = {}
    for c in sch.columns:
        vals = [r[c.name] for r in rows]
        if c.kind in ("ts", "ts?", "int"):
            cols[c.name] = np.asarray(vals, dtype=np.int64)
        elif c.kind in ("float", "float?"):
            cols[c.name] = np.asarray(vals, dtype=np.float64)
        else:
            cols[c.name] = np.asarray(vals, dtype=object)
    return sort_table(Table(sch, cols))


def _new_bars(rng, iid, cal_open, cal_event, ks, price, volume):
    out = []
    c_prev = price
    for k in ks:
        o = c_prev * math.exp(rng.normal(0, 0.01))
        c = o * math.exp(rng.normal(0, 0.02))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.005)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.005)))
        v = float(round(volume * math.exp(rng.normal(0, 0.5))))
        out.append(
            {
                "instrument_id": iid,
                "ts_open": int(cal_open[k]),
                "ts_event": int(cal_event[k]),
                "ts_avail": int(cal_event[k]),
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": v,
            }
        )
        c_prev = c
    return out


def poison_store(store: Store, t: int, seed: int, check: bool = True) -> Store:
    rng = stream(int(seed), "audit.poison")
    p = store.panel()
    cal_o, cal_e = p.ts_open, p.ts_event
    T = len(cal_e)
    kt = int(np.searchsorted(cal_e, t, side="right")) - 1
    ins = _rows(store.tables["instruments"])
    bars_by: dict[str, list[dict]] = {}
    for r in _rows(store.tables["bars"]):
        bars_by.setdefault(r["instrument_id"], []).append(r)
    actions = _rows(store.tables["corporate_actions"])
    series = _rows(store.tables["series"])

    removed: set[str] = set()
    new_ins: list[dict] = []
    for r in ins:
        iid = r["instrument_id"]
        bl = bars_by.get(iid, [])
        if r["ts_list"] > t:
            if rng.random() < 0.3:
                removed.add(iid)
                continue
        if r["ts_delist"] != NO_TS and r["ts_delist"] > t and bl:
            if rng.random() < 0.5:  # cancel the future delisting: bars continue
                last_k = int(np.searchsorted(cal_e, bl[-1]["ts_event"]))
                ks = list(range(last_k + 1, T))
                bl.extend(
                    _new_bars(
                        rng, iid, cal_o, cal_e, ks, bl[-1]["close"], max(1.0, bl[-1]["volume"])
                    )
                )
                r["ts_delist"] = NO_TS
                r["delist_return"] = math.nan
        elif r["ts_delist"] == NO_TS and bl and rng.random() < 0.3:
            ks = [int(np.searchsorted(cal_e, b["ts_event"])) for b in bl]
            cand = [k for k in ks if k > kt]
            if cand:  # add a delisting at least one bar after t
                L = int(cand[int(rng.integers(0, len(cand)))])
                bl[:] = [b for b in bl if b["ts_event"] <= cal_e[L]]
                r["ts_delist"] = int(cal_e[L])
                r["delist_return"] = float(rng.uniform(-0.9, 0.2))
    # new instruments listing after t
    if kt + 2 < T:
        for _ in range(int(rng.integers(1, 3))):
            base = ins[int(rng.integers(0, len(ins)))]["instrument_id"] if ins else "X"
            iid = f"{base}p{int(rng.integers(0, 1000))}"
            if iid in bars_by or any(x["instrument_id"] == iid for x in new_ins):
                continue
            k0 = int(rng.integers(kt + 1, T))
            bars_by[iid] = _new_bars(
                rng, iid, cal_o, cal_e, list(range(k0, T)), float(rng.uniform(5, 200)), 1e5
            )
            new_ins.append(
                {
                    "instrument_id": iid,
                    "symbol": iid,
                    "asset_class": "equity",
                    "currency": "USD",
                    "ts_list": int(cal_e[k0]),
                    "ts_delist": NO_TS,
                    "delist_return": math.nan,
                    "lot_size": 0.0,
                    "tick_size": 0.01,
                    "sector": "",
                }
            )
    ins = [r for r in ins if r["instrument_id"] not in removed] + new_ins
    # bars after t: random price factors and volumes
    all_bars = []
    for r in ins:
        for b in bars_by.get(r["instrument_id"], []):
            if b["ts_event"] > t:
                f = math.exp(rng.uniform(math.log(0.1), math.log(10.0)))
                b = dict(b)
                for key in ("open", "high", "low", "close"):
                    b[key] = b[key] * f
                b["high"] = max(b["high"], b["open"], b["close"])
                b["low"] = min(b["low"], b["open"], b["close"])
                b["volume"] = (
                    0.0
                    if rng.random() < 0.03
                    else float(round(max(1.0, b["volume"]) * math.exp(rng.normal(0, 1.0))))
                )
            all_bars.append(b)
    # corporate actions
    alive_ids = {r["instrument_id"] for r in ins}
    last_ev = {
        iid: (bars_by[iid][-1]["ts_event"] if bars_by.get(iid) else NO_TS) for iid in alive_ids
    }
    acts = []
    keys = set()
    for a in actions:
        if a["instrument_id"] not in alive_ids:
            continue
        if a["ts_avail"] > t:
            if rng.random() < 0.25 or a["ts_ex"] > last_ev[a["instrument_id"]]:
                continue
            a = dict(a)
            if a["action"] == "split":
                a["value"] = 3.0 if a["value"] == 2.0 else 2.0
            else:
                a["value"] = round(a["value"] * float(rng.uniform(0.5, 2.0)), 4) or 0.01
        acts.append(a)
        keys.add((a["instrument_id"], a["ts_ex"], a["action"]))
    for r in ins:
        iid = r["instrument_id"]
        fut = [b for b in bars_by.get(iid, []) if b["ts_event"] > t]
        if len(fut) >= 2 and rng.random() < 0.5:
            j = int(rng.integers(1, len(fut)))
            ex = fut[j]["ts_open"]
            kev = [int(cal_e[m]) for m in range(kt + 1, T) if cal_e[m] < ex]
            av = kev[int(rng.integers(0, len(kev)))] if kev else ex
            key = (iid, ex, "split")
            if key not in keys and av > t:
                acts.append(
                    {
                        "instrument_id": iid,
                        "action": "split",
                        "ts_ex": ex,
                        "ts_avail": av,
                        "value": float(rng.choice([2.0, 3.0])),
                    }
                )
                keys.add(key)
    # series
    out_series = []
    by_period: dict[tuple, list[dict]] = {}
    for s in series:
        iid = s["series_id"].split(".")[0]
        if iid in removed:
            continue
        by_period.setdefault((s["series_id"], s["ts_event"]), []).append(s)
    after = [int(x) for x in cal_e[kt + 1 :]]
    for (sid, te), rows in by_period.items():
        known = [dict(x) for x in rows if x["ts_avail"] <= t]
        later = [dict(x) for x in rows if x["ts_avail"] > t]
        if known:
            if later:
                if rng.random() < 0.5:
                    for x in later:
                        x["value"] = float(rng.normal()) * 3.0
                    known.extend(later)
            elif after and rng.random() < 0.5:
                av = after[int(rng.integers(0, min(len(after), 20)))]
                known.append(
                    {
                        "series_id": sid,
                        "ts_event": te,
                        "ts_avail": av,
                        "vintage": known[-1]["vintage"] + 1,
                        "value": float(rng.normal()) * 3.0,
                    }
                )
            out_series.extend(known)
        else:
            for x in later:
                x["value"] = float(rng.normal()) * 3.0
            out_series.extend(later)
    meta = copy.deepcopy(store.meta)
    meta["seed"] = int(rng.integers(1, 2**31))
    gen = meta.get("generator") or {}
    if isinstance(gen, dict):
        params = dict(gen.get("parameters") or {})
        params["ic"] = float(rng.uniform(0.05, 0.5))
        params["n_instruments"] = len(ins)
        gen = dict(gen)
        gen["parameters"] = params
        meta["generator"] = gen
    meta.pop("config_hash", None)
    bars_t = _table("bars", all_bars)
    acts_t = _table("corporate_actions", acts)
    tables = {
        "instruments": _table("instruments", ins),
        "bars": bars_t,
        "corporate_actions": acts_t,
        "series": _table("series", out_series),
        "liquidity": liquidity_table(bars_t, acts_t),
    }
    if check:
        for name in ("instruments", "bars", "corporate_actions", "series"):
            validate_table(tables[name])
        _check_known_unchanged(store, tables, t)
    return Store(tables, meta)


def _known_bytes(table: Table, t: int, col: str) -> list[tuple]:
    idx = np.flatnonzero(table[col] <= t)
    names = table.schema.names
    return sorted(
        tuple(
            table[n][i] if not isinstance(table[n][i], float) else float(table[n][i]).hex()
            for n in names
        )
        for i in idx.tolist()
    )


def _check_known_unchanged(store: Store, tables: dict, t: int) -> None:
    for name, col in (
        ("bars", "ts_avail"),
        ("series", "ts_avail"),
        ("corporate_actions", "ts_avail"),
        ("liquidity", "ts_avail"),
    ):
        a = _known_bytes(store.tables[name], t, col)
        b = _known_bytes(tables[name], t, col)
        if a != b:
            raise AssertionError(f"poison changed rows of {name} known at t")
