"""The point-in-time store.

A store is a directory of Parquet files (zstd, instants as ``timestamp[us, tz=UTC]``):
``instruments``, ``bars``, ``corporate_actions``, ``series``, ``liquidity``, and a ``store.json``
manifest (schema versions, per-file ``content_sha256``, calendar, ``ppy``, generator or source,
contract version). In memory a store holds the same tables as NumPy columns.

Every query takes a required keyword argument ``knowledge`` (an instant) and has no default,
and no query returns a row with ``ts_avail`` after ``knowledge``. Strategies never receive a
store; the harness builds their :class:`~quant_research_engine.context.DecisionContext`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_engine import QC_VERSION
from quant_research_engine.data.csvio import NO_TS, Table, empty_table, validate_table
from quant_research_engine.data.manifest import read_json, sha256_file, write_json
from quant_research_engine.data.schemas import SCHEMA_VERSION, SCHEMAS
from quant_research_engine.store.adjust import (
    bar_events,
    bar_returns,
    split_adjusted_close_asof,
    total_return_asof,
)
from quant_research_engine.store.asof import asof_vintage

STORE_TABLES = ("instruments", "bars", "corporate_actions", "series", "liquidity")


@dataclass
class Panel:
    """Calendar-aligned arrays of a store (harness-side; strategies never see a Panel).

    Rows are the sessions of the store's calendar (the sorted union of bar ``ts_event``),
    columns the instruments in ``instrument_id`` order. Prices are raw; NaN where an instrument
    has no bar. ``ratio``/``div`` are the split ratio and dividend per pre-split share effective
    in a bar; ``ret`` the returns of 2.5; ``adv``/``sigma`` the liquidity row of the bar (NaN if
    none) and ``liq_avail`` its ``ts_avail``.
    """

    ids: list[str]
    ts_open: np.ndarray
    ts_event: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    avail: np.ndarray
    has_bar: np.ndarray
    ratio: np.ndarray
    div: np.ndarray
    ret: np.ndarray
    adv: np.ndarray
    sigma: np.ndarray
    liq_avail: np.ndarray
    ts_list: np.ndarray
    ts_delist: np.ndarray
    delist_return: np.ndarray
    lot_size: np.ndarray
    first_bar: np.ndarray
    last_bar: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape


class Store:
    """In-memory view of a store directory (or of a store built in memory)."""

    def __init__(self, tables: dict[str, Table], meta: dict, path: str | None = None):
        for name in STORE_TABLES:
            if name not in tables:
                tables[name] = empty_table(SCHEMAS[name])
        self.tables = tables
        self.meta = meta
        self.path = path
        self._index()
        self._panel: Panel | None = None

    # ------------------------------------------------------------------ construction

    def _index(self) -> None:
        ins = self.tables["instruments"]
        self.ids: list[str] = [str(x) for x in ins["instrument_id"].tolist()]
        self._pos = {k: i for i, k in enumerate(self.ids)}
        bars = self.tables["bars"]
        iid = bars["instrument_id"]
        self._bar_slices: dict[str, tuple[int, int]] = {}
        if len(bars):
            starts = np.flatnonzero(np.r_[True, iid[1:] != iid[:-1]])
            ends = np.r_[starts[1:], len(bars)]
            for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
                self._bar_slices[str(iid[s])] = (s, e)
        ca = self.tables["corporate_actions"]
        self._actions: dict[str, list[tuple[str, int, float, int]]] = {}
        for i in range(len(ca)):
            self._actions.setdefault(str(ca["instrument_id"][i]), []).append(
                (
                    str(ca["action"][i]),
                    int(ca["ts_ex"][i]),
                    float(ca["value"][i]),
                    int(ca["ts_avail"][i]),
                )
            )
        ser = self.tables["series"]
        self._series_slices: dict[str, tuple[int, int]] = {}
        sid = ser["series_id"]
        if len(ser):
            starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
            ends = np.r_[starts[1:], len(ser)]
            for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
                self._series_slices[str(sid[s])] = (s, e)

    @property
    def calendar_name(self) -> str:
        return str(self.meta.get("calendar", ""))

    @property
    def ppy(self) -> int:
        return int(self.meta.get("ppy", 252))

    def series_ids(self) -> list[str]:
        return sorted(self._series_slices)

    # ------------------------------------------------------------------ queries

    def instruments_at(self, knowledge: int) -> list[str]:
        """Universe at ``knowledge`` (contract 3.3): listed and not yet delisted."""
        t = int(knowledge)
        ins = self.tables["instruments"]
        dl = ins["ts_delist"]
        ok = (ins["ts_list"] <= t) & ((dl == NO_TS) | (dl > t))
        return [self.ids[i] for i in np.flatnonzero(ok).tolist()]

    def instruments(self, *, knowledge: int) -> Table:
        """Instruments known at ``knowledge``; delisting fields read as empty before ts_delist."""
        t = int(knowledge)
        ins = self.tables["instruments"]
        keep = np.flatnonzero(ins["ts_list"] <= t)
        out = ins.take(keep)
        cols = dict(out.columns)
        dl = cols["ts_delist"].copy()
        dr = cols["delist_return"].copy()
        hide = (dl != NO_TS) & (dl > t)
        dl[hide] = NO_TS
        dr[hide] = np.nan
        cols["ts_delist"], cols["delist_return"] = dl, dr
        return Table(out.schema, cols)

    def _known_bar_idx(self, iid: str, t: int) -> np.ndarray:
        s = self._bar_slices.get(iid)
        if s is None:
            return np.zeros(0, dtype=np.int64)
        a, b = s
        avail = self.tables["bars"]["ts_avail"][a:b]
        return a + np.flatnonzero(avail <= t)

    def bars(self, instrument_ids: list[str], n: int, *, knowledge: int) -> dict[str, Table]:
        """The last ``n`` known bars of each instrument (unknown ids give empty tables)."""
        t = int(knowledge)
        n = int(n)
        out: dict[str, Table] = {}
        bars = self.tables["bars"]
        for iid in instrument_ids:
            idx = self._known_bar_idx(str(iid), t)
            idx = idx[max(0, len(idx) - n) :] if n > 0 else idx[:0]
            out[str(iid)] = bars.take(idx)
        return out

    def corporate_actions(self, *, knowledge: int) -> Table:
        ca = self.tables["corporate_actions"]
        return ca.take(np.flatnonzero(ca["ts_avail"] <= int(knowledge)))

    def series(self, series_id: str, *, knowledge: int) -> dict[str, np.ndarray]:
        """As-of rows of one series (largest known vintage per period), by ts_event."""
        s = self._series_slices.get(str(series_id))
        ser = self.tables["series"]
        if s is None:
            sub = ser.take(np.zeros(0, dtype=np.int64))
        else:
            sub = ser.take(np.arange(s[0], s[1]))
        idx = asof_vintage(sub.columns, int(knowledge))
        return {k: sub[k][idx] for k in ("ts_event", "ts_avail", "vintage", "value")}

    def liquidity(self, *, knowledge: int) -> Table:
        """Per instrument, the latest liquidity row with ``ts_avail <= knowledge``."""
        liq = self.tables["liquidity"]
        known = np.flatnonzero(liq["ts_avail"] <= int(knowledge))
        if known.size == 0:
            return liq.take(known)
        iid = liq["instrument_id"][known]
        te = liq["ts_event"][known]
        order = sorted(range(known.size), key=lambda i: (iid[i], te[i]))
        last: dict[str, int] = {}
        for i in order:
            last[str(iid[i])] = int(known[i])
        return liq.take(np.asarray([last[k] for k in sorted(last)], dtype=np.int64))

    def instrument_actions(self, iid: str, *, knowledge: int) -> list[tuple[str, int, float]]:
        return [
            (a, ex, v) for a, ex, v, av in self._actions.get(str(iid), []) if av <= int(knowledge)
        ]

    def adjusted_close(
        self, instrument_ids: list[str], n: int, *, knowledge: int, total_return: bool = False
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """(ts_event, adjusted close) of the last ``n`` known bars, adjusted as of ``knowledge``
        (contract 3.2): split-adjusted, or the total-return series when ``total_return``."""
        t = int(knowledge)
        bars = self.tables["bars"]
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for iid in instrument_ids:
            idx = self._known_bar_idx(str(iid), t)
            close = bars["close"][idx]
            to = bars["ts_open"][idx]
            acts = self.instrument_actions(str(iid), knowledge=t)
            splits = [(ex, v) for a, ex, v in acts if a == "split"]
            adj = split_adjusted_close_asof(close, to, splits, t)
            if total_return:
                ratio, div = bar_events(to, acts)
                adj = total_return_asof(adj, bar_returns(close, ratio, div))
            k = max(0, len(idx) - int(n)) if n > 0 else len(idx)
            out[str(iid)] = (bars["ts_event"][idx][k:], adj[k:])
        return out

    # ------------------------------------------------------------------ panel (harness)

    def panel(self) -> Panel:
        if self._panel is None:
            self._panel = build_panel(self)
        return self._panel


def build_panel(store: Store) -> Panel:
    bars = store.tables["bars"]
    ids = store.ids
    n_ins = len(ids)
    cal_ev = np.unique(bars["ts_event"]) if len(bars) else np.zeros(0, dtype=np.int64)
    T = len(cal_ev)
    cal_op = np.zeros(T, dtype=np.int64)
    shape = (T, n_ins)
    nan = np.full(shape, np.nan)
    p = {k: nan.copy() for k in ("open", "high", "low", "close", "volume", "ret", "adv", "sigma")}
    ratio = np.ones(shape)
    div = np.zeros(shape)
    avail = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
    liq_avail = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
    has = np.zeros(shape, dtype=bool)
    first = np.full(n_ins, -1, dtype=np.int64)
    last = np.full(n_ins, -1, dtype=np.int64)
    seen_open = np.zeros(T, dtype=bool)
    for j, iid in enumerate(ids):
        s = store._bar_slices.get(iid)
        if s is None:
            continue
        a, b = s
        rows = np.searchsorted(cal_ev, bars["ts_event"][a:b])
        for k in ("open", "high", "low", "close", "volume"):
            p[k][rows, j] = bars[k][a:b]
        avail[rows, j] = bars["ts_avail"][a:b]
        has[rows, j] = True
        new = ~seen_open[rows]
        cal_op[rows[new]] = bars["ts_open"][a:b][new]
        seen_open[rows] = True
        first[j], last[j] = rows[0], rows[-1]
        acts = [(x[0], x[1], x[2]) for x in store._actions.get(iid, [])]
        r_, d_ = bar_events(bars["ts_open"][a:b], acts)
        ratio[rows, j] = r_
        div[rows, j] = d_
        p["ret"][rows, j] = bar_returns(bars["close"][a:b], r_, d_)
    liq = store.tables["liquidity"]
    if len(liq):
        cols = np.asarray(
            [store._pos[str(x)] for x in liq["instrument_id"].tolist()], dtype=np.int64
        )
        rows = np.searchsorted(cal_ev, liq["ts_event"])
        ok = (rows < T) & (cal_ev[np.minimum(rows, T - 1)] == liq["ts_event"])
        p["adv"][rows[ok], cols[ok]] = liq["adv_shares"][ok]
        p["sigma"][rows[ok], cols[ok]] = liq["sigma_bar"][ok]
        liq_avail[rows[ok], cols[ok]] = liq["ts_avail"][ok]
    ins = store.tables["instruments"]
    return Panel(
        ids=list(ids),
        ts_open=cal_op,
        ts_event=cal_ev.astype(np.int64),
        open=p["open"],
        high=p["high"],
        low=p["low"],
        close=p["close"],
        volume=p["volume"],
        avail=avail,
        has_bar=has,
        ratio=ratio,
        div=div,
        ret=p["ret"],
        adv=p["adv"],
        sigma=p["sigma"],
        liq_avail=liq_avail,
        ts_list=ins["ts_list"].copy(),
        ts_delist=ins["ts_delist"].copy(),
        delist_return=ins["delist_return"].copy(),
        lot_size=ins["lot_size"].copy(),
        first_bar=first,
        last_bar=last,
    )


# ---------------------------------------------------------------------- Parquet I/O


def _arrow_type(kind: str):
    if kind in ("ts", "ts?"):
        return pa.timestamp("us", tz="UTC")
    if kind in ("float", "float?"):
        return pa.float64()
    if kind == "int":
        return pa.int64()
    if kind == "bool":
        return pa.bool_()
    return pa.string()


def table_to_arrow(table: Table) -> pa.Table:
    arrays, names = [], []
    for c in table.schema.columns:
        v = table[c.name]
        if c.kind == "ts?":
            arr = pa.array(v, type=pa.int64(), mask=(v == NO_TS)).cast(_arrow_type(c.kind))
        elif c.kind == "ts":
            arr = pa.array(v, type=pa.int64()).cast(_arrow_type(c.kind))
        elif c.kind == "float?":
            arr = pa.array(v, type=pa.float64(), mask=np.isnan(v))
        elif c.kind in ("float", "int", "bool"):
            arr = pa.array(v, type=_arrow_type(c.kind))
        else:
            arr = pa.array([str(x) for x in v.tolist()], type=pa.string())
        arrays.append(arr)
        names.append(c.name)
    return pa.Table.from_arrays(arrays, names=names)


def arrow_to_table(at: pa.Table, schema_name: str) -> Table:
    schema = SCHEMAS[schema_name]
    cols: dict[str, np.ndarray] = {}
    for c in schema.columns:
        col = at.column(c.name).combine_chunks()
        if c.kind in ("ts", "ts?"):
            ints = col.cast(pa.int64())
            cols[c.name] = ints.fill_null(NO_TS).to_numpy(zero_copy_only=False).astype(np.int64)
        elif c.kind in ("float", "float?"):
            cols[c.name] = col.fill_null(np.nan).to_numpy(zero_copy_only=False).astype(np.float64)
        elif c.kind == "int":
            cols[c.name] = col.to_numpy(zero_copy_only=False).astype(np.int64)
        elif c.kind == "bool":
            cols[c.name] = col.to_numpy(zero_copy_only=False).astype(bool)
        else:
            cols[c.name] = np.asarray(col.to_pylist(), dtype=object)
    return Table(schema, cols)


def write_store(path: str, store: Store) -> dict:
    """Write the tables as Parquet files and ``store.json``; returns the manifest."""
    os.makedirs(path, exist_ok=True)
    files = {}
    for name in STORE_TABLES:
        t = store.tables[name]
        validate_table(t)
        fp = os.path.join(path, f"{name}.parquet")
        pq.write_table(table_to_arrow(t), fp, compression="zstd", use_dictionary=False)
        files[name] = {"content_sha256": sha256_file(fp), "row_count": len(t)}
    manifest = dict(store.meta)
    manifest.update({"schema_version": SCHEMA_VERSION, "qc_version": QC_VERSION, "files": files})
    write_json(os.path.join(path, "store.json"), manifest)
    store.path = path
    return manifest


def read_store(path: str, verify: bool = True) -> Store:
    manifest = read_json(os.path.join(path, "store.json"))
    if manifest.get("qc_version") != QC_VERSION or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported store version")
    tables = {}
    for name in STORE_TABLES:
        fp = os.path.join(path, f"{name}.parquet")
        if verify and sha256_file(fp) != manifest["files"][name]["content_sha256"]:
            raise ValueError(f"{name}.parquet does not match store.json")
        tables[name] = arrow_to_table(pq.read_table(fp), name)
    meta = {k: v for k, v in manifest.items() if k not in ("files", "schema_version", "qc_version")}
    return Store(tables, meta, path=path)


def cut_store(store: Store, t: int) -> Store:
    """The store as known at ``t``: rows with ``ts_avail <= t``; instruments listed by ``t``, with
    ``ts_delist`` and ``delist_return`` hidden until ``ts_delist`` (contract 3.1)."""
    t = int(t)
    tables = {}
    for name in ("bars", "corporate_actions", "series", "liquidity"):
        tab = store.tables[name]
        tables[name] = tab.take(np.flatnonzero(tab["ts_avail"] <= t))
    tables["instruments"] = store.instruments(knowledge=t)
    return Store(tables, dict(store.meta))
