"""DecisionContext: the only object a strategy receives (guard G1, context truncation).

The context is built by the harness-side :class:`ContextBuilder` from copies of the rows known
at the decision instant ``t``: every array it hands out is a read-only copy cut at the knowledge
boundary (``ts_avail <= t``), with no ``.base`` pointing to a longer array, and the context holds
no reference to the store, the panel, or any array that extends past ``t``. Within the history
window a strategy declares (``history_bars``), accessors return the last ``n`` known rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from quant_research_engine.rng import stream
from quant_research_engine.store.adjust import split_factor_asof
from quant_research_engine.store.asof import asof_vintage

BAR_FIELDS = ("open", "high", "low", "close", "volume")


def _ro(a: np.ndarray) -> np.ndarray:
    out = np.array(a, copy=True, order="C")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class OpenOrder:
    order_id: str
    instrument_id: str
    quantity: float
    tif: str
    ts_submit: int


@dataclass(frozen=True)
class PortfolioView:
    """Cash, equity, and positions after all fills up to the decision instant."""

    cash: float
    equity: float
    positions: MappingProxyType  # instrument_id -> quantity (nonzero only)

    def quantity(self, instrument_id: str) -> float:
        return float(self.positions.get(instrument_id, 0.0))


def empty_portfolio(cash: float) -> PortfolioView:
    return PortfolioView(float(cash), float(cash), MappingProxyType({}))


class DecisionContext:
    """Knowledge-bounded accessors at decision instant ``t`` (instruments in id order)."""

    __slots__ = (
        "_t",
        "_ids",
        "_ts_event",
        "_ts_open",
        "_fields",
        "_ret",
        "_splits",
        "_series",
        "_adv",
        "_sigma",
        "_portfolio",
        "_orders",
        "_seed",
        "_strategy",
    )

    def __init__(
        self,
        t,
        ids,
        ts_event,
        ts_open,
        fields,
        ret,
        splits,
        series,
        adv,
        sigma,
        portfolio,
        open_orders,
        strategy_seed,
        strategy_name,
    ):
        self._t = int(t)
        self._ids = tuple(ids)
        self._ts_event = ts_event
        self._ts_open = ts_open
        self._fields = MappingProxyType(dict(fields))
        self._ret = ret
        self._splits = MappingProxyType(dict(splits))
        self._series = MappingProxyType(dict(series))
        self._adv = adv
        self._sigma = sigma
        self._portfolio = portfolio
        self._orders = tuple(open_orders)
        self._seed = int(strategy_seed)
        self._strategy = str(strategy_name)

    # ----------------------------------------------------------------- accessors

    @property
    def t(self) -> int:
        """The decision instant (microseconds since the epoch, UTC)."""
        return self._t

    def universe(self) -> tuple[str, ...]:
        """Instruments listed and not delisted at ``t`` (including later delistings)."""
        return self._ids

    def _cut(self, n: int) -> slice:
        rows = len(self._ts_event)
        n = int(n)
        if n <= 0:
            return slice(rows, rows)
        return slice(max(0, rows - n), rows)

    def bars(self, field: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(values ``n x N``, ts_event ``n``) of the last ``n`` known sessions; NaN where an
        instrument has no known bar. ``field`` is open, high, low, close, or volume."""
        if field not in self._fields:
            raise KeyError(f"unknown bar field {field!r}")
        s = self._cut(n)
        return _ro(self._fields[field][s]), _ro(self._ts_event[s])

    def returns(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Simple total returns of the shared contract (2.5) of the last ``n`` known bars."""
        s = self._cut(n)
        return _ro(self._ret[s]), _ro(self._ts_event[s])

    def adjusted_close(self, n: int, total_return: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Closes adjusted as of ``t`` (splits with ts_ex <= t only; or the total-return series
        of the known bars, scaled to the last split-adjusted close)."""
        close = self._fields["close"]
        out = np.full(close.shape, np.nan)
        for j, iid in enumerate(self._ids):
            col = close[:, j]
            ok = np.flatnonzero(~np.isnan(col))
            if ok.size == 0:
                continue
            f = split_factor_asof(
                ok.size, self._ts_open[ok], list(self._splits.get(iid, ())), self._t
            )
            adj = col[ok] / f
            if total_return:
                r = self._ret[ok, j]
                growth = np.ones(ok.size)
                if ok.size > 1:
                    growth[1:] = np.cumprod(1.0 + np.nan_to_num(r[1:]))
                adj = growth * (adj[-1] / growth[-1])
            out[ok, j] = adj
        s = self._cut(n)
        return _ro(out[s]), _ro(self._ts_event[s])

    def series(self, series_id: str) -> tuple[np.ndarray, np.ndarray]:
        """(ts_event, value) of a declared series as of ``t`` (largest known vintage)."""
        if series_id not in self._series:
            raise KeyError(f"series {series_id!r} is not declared by the strategy or unknown")
        te, val = self._series[series_id]
        return _ro(te), _ro(val)

    def liquidity(self) -> tuple[np.ndarray, np.ndarray]:
        """(adv_shares, sigma_bar) of the latest known liquidity row per instrument (NaN: none)."""
        return _ro(self._adv), _ro(self._sigma)

    def portfolio(self) -> PortfolioView:
        return self._portfolio

    def open_orders(self) -> tuple[OpenOrder, ...]:
        return self._orders

    def rng(self, name: str) -> np.random.Generator:
        """A fresh generator for stream ``strategy.<strategy name>.<name>``."""
        return stream(self._seed, f"strategy.{self._strategy}.{name}")


class ContextBuilder:
    """Harness-side factory of contexts for one store and one strategy.

    Holds the store's panel; a context it builds holds only copies of rows known at ``t``.
    """

    def __init__(
        self,
        store,
        strategy_name: str,
        strategy_seed: int = 0,
        history_bars: int | None = None,
        series_suffixes: tuple[str, ...] = (),
    ):
        self._store = store
        self._p = store.panel()
        self._name = strategy_name
        self._seed = int(strategy_seed)
        self._hist = history_bars
        self._suffixes = tuple(series_suffixes)
        p = self._p
        self._splits: dict[str, list[tuple[int, float]]] = {}
        for iid, acts in store._actions.items():
            self._splits[iid] = sorted((ex, v) for a, ex, v, av in acts if a == "split")
        # liquidity: index of the latest row at or before each session
        T, N = p.shape
        idx = np.where(~np.isnan(p.adv), np.arange(T)[:, None], -1)
        self._liq_idx = np.maximum.accumulate(idx, axis=0) if T else idx
        # per-instrument series rows (declared suffixes only)
        self._ser: dict[str, tuple] = {}
        ser = store.tables["series"]
        for iid in p.ids:
            for suf in self._suffixes:
                sid = f"{iid}{suf}"
                sl = store._series_slices.get(sid)
                if sl is None:
                    continue
                a, b = sl
                cols = {k: ser[k][a:b] for k in ("ts_event", "ts_avail", "vintage", "value")}
                fast = bool(np.all(cols["vintage"] == 0) and np.all(np.diff(cols["ts_avail"]) >= 0))
                self._ser[sid] = (cols, fast)

    def _series_asof(self, sid: str, t: int):
        cols, fast = self._ser[sid]
        if fast:
            j = int(np.searchsorted(cols["ts_avail"], t, side="right"))
            return cols["ts_event"][:j].copy(), cols["value"][:j].copy()
        rows = dict(cols)
        rows["series_id"] = np.full(len(cols["ts_event"]), sid, dtype=object)
        idx = asof_vintage(rows, t)
        return cols["ts_event"][idx].copy(), cols["value"][idx].copy()

    def build(self, t: int, portfolio: PortfolioView, open_orders=()) -> DecisionContext:
        p = self._p
        t = int(t)
        ids = self._store.instruments_at(t)
        pos = self._store._pos
        cols = np.asarray([pos[i] for i in ids], dtype=np.int64) if ids else np.zeros(0, np.int64)
        end = int(np.searchsorted(p.ts_event, t, side="right"))
        start = 0 if self._hist is None else max(0, end - int(self._hist))
        block = np.s_[start:end]
        unknown = p.avail[block][:, cols] > t
        has_unknown = bool(unknown.any())
        fields = {}
        for f in BAR_FIELDS:
            a = np.array(getattr(p, f)[block][:, cols], copy=True)
            if has_unknown:
                a[unknown] = np.nan
            a.setflags(write=False)
            fields[f] = a
        ret = np.array(p.ret[block][:, cols], copy=True)
        if has_unknown:
            ret[unknown] = np.nan
        ret.setflags(write=False)
        ts_event = _ro(p.ts_event[block])
        ts_open = _ro(p.ts_open[block])
        splits = {i: tuple((ex, v) for ex, v in self._splits.get(i, ()) if ex <= t) for i in ids}
        series = {}
        for i in ids:
            for suf in self._suffixes:
                sid = f"{i}{suf}"
                if sid in self._ser:
                    te, val = self._series_asof(sid, t)
                    te.setflags(write=False)
                    val.setflags(write=False)
                    series[sid] = (te, val)
        adv = np.full(len(ids), np.nan)
        sig = np.full(len(ids), np.nan)
        if end > 0:
            for j, c in enumerate(cols.tolist()):
                r = int(self._liq_idx[end - 1, c])
                while r >= 0 and p.liq_avail[r, c] > t:
                    r = int(self._liq_idx[r - 1, c]) if r > 0 else -1
                if r >= 0:
                    adv[j], sig[j] = p.adv[r, c], p.sigma[r, c]
        adv.setflags(write=False)
        sig.setflags(write=False)
        return DecisionContext(
            t,
            ids,
            ts_event,
            ts_open,
            fields,
            ret,
            splits,
            series,
            adv,
            sig,
            portfolio,
            open_orders,
            self._seed,
            self._name,
        )


def is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))
