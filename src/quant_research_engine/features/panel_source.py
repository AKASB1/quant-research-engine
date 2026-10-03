"""Inputs of panel features, with availability tracking for the availability audit (guard G4).

Every accessor returns a (T, N) panel aligned on the store's calendar (instruments in id order)
and, when tracking is on, records the panel of ``ts_avail`` of the input rows it returned. The
audit derives a feature value's availability as the latest ``ts_avail`` of the inputs in its
lookback window and fails a feature that declares an earlier availability.
"""

from __future__ import annotations

import numpy as np

NEVER = np.iinfo(np.int64).max


class PanelSource:
    def __init__(self, store, track: bool = False):
        self.store = store
        self.p = store.panel()
        self.ids = list(self.p.ids)
        self.ts_event = self.p.ts_event
        self.T, self.N = self.p.shape
        self.track = track
        self.used: list[np.ndarray] = []
        self._bar_avail = np.where(self.p.has_bar, self.p.avail, 0)

    def _reg(self, avail: np.ndarray) -> None:
        if self.track:
            self.used.append(avail)

    def bars(self, field: str) -> np.ndarray:
        self._reg(self._bar_avail)
        p = self.p
        fields = {
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.volume,
        }
        return fields[field].copy()

    def returns(self) -> np.ndarray:
        self._reg(self._bar_avail)
        return self.p.ret.copy()

    def universe_mask(self) -> np.ndarray:
        """(T, N): listed and not delisted at each session's close (contract 3.3)."""
        te = self.ts_event[:, None]
        dl = self.p.ts_delist[None, :]
        return (self.p.ts_list[None, :] <= te) & ((dl == -1) | (dl > te))

    def _series_rows(self, sid: str):
        st = self.store
        sl = st._series_slices.get(sid)
        if sl is None:
            return None
        ser = st.tables["series"]
        a, b = sl
        return {k: ser[k][a:b] for k in ("ts_event", "ts_avail", "vintage", "value")}

    def series_asof(self, suffix: str) -> np.ndarray:
        """At each session close, the latest known value of ``<id><suffix>``: the period with the
        largest ts_event among those with a vintage known then, at its largest known vintage."""
        out = np.full((self.T, self.N), np.nan)
        av = np.zeros((self.T, self.N), dtype=np.int64)
        te_cal = self.ts_event
        for j, iid in enumerate(self.ids):
            rows = self._series_rows(iid + suffix)
            if rows is None or len(rows["ts_event"]) == 0:
                continue
            ta = rows["ts_avail"]
            if (
                np.all(rows["vintage"] == 0)
                and np.all(np.diff(rows["ts_event"]) > 0)
                and np.all(np.diff(ta) >= 0)
            ):
                # one vintage per period, released in period order: forward fill by release
                k = np.searchsorted(te_cal, ta, side="left")
                ok = k < self.T
                col = np.full(self.T, -1, dtype=np.int64)
                col[k[ok]] = np.flatnonzero(ok)
                col = np.maximum.accumulate(col)
                has = col >= 0
                out[has, j] = rows["value"][col[has]]
                av[has, j] = ta[col[has]]
                continue
            order = np.argsort(ta, kind="stable")
            state: dict[int, tuple[int, float, int]] = {}
            ptr = 0
            for kk in range(self.T):
                t = te_cal[kk]
                while ptr < len(order) and ta[order[ptr]] <= t:
                    r = order[ptr]
                    e = int(rows["ts_event"][r])
                    v = int(rows["vintage"][r])
                    if e not in state or v > state[e][0]:
                        state[e] = (v, float(rows["value"][r]), int(ta[r]))
                    ptr += 1
                if state:
                    e = max(state)
                    out[kk, j] = state[e][1]
                    av[kk, j] = state[e][2]
        self._reg(av)
        return out

    def series_by_event(self, suffix: str, vintage: int = 0) -> np.ndarray:
        """The mistake of canary C6: each period's value placed at its period end (ts_event),
        regardless of when it was released, forward-filled."""
        out = np.full((self.T, self.N), np.nan)
        av = np.zeros((self.T, self.N), dtype=np.int64)
        for j, iid in enumerate(self.ids):
            rows = self._series_rows(iid + suffix)
            if rows is None:
                continue
            sel = np.flatnonzero(rows["vintage"] == vintage)
            k = np.searchsorted(self.ts_event, rows["ts_event"][sel], side="left")
            ok = k < self.T
            col = np.full(self.T, -1, dtype=np.int64)
            col[k[ok]] = sel[ok]
            col = np.maximum.accumulate(col)
            has = col >= 0
            out[has, j] = rows["value"][col[has]]
            av[has, j] = rows["ts_avail"][col[has]]
        self._reg(av)
        return out

    def declared_bar_avail(self) -> np.ndarray:
        """Default declared availability of a value computed at bar k: the bar's ts_avail."""
        return np.where(self.p.has_bar, self.p.avail, NEVER)


def derived_availability(used: list[np.ndarray], lookback: int) -> np.ndarray:
    """Latest ts_avail of the inputs in the window [k - lookback, k] (per instrument)."""
    if not used:
        raise ValueError("feature registered no inputs")
    m = used[0].copy()
    for u in used[1:]:
        m = np.maximum(m, u)
    out = m.copy()
    for lag in range(1, lookback + 1):
        out[lag:] = np.maximum(out[lag:], m[:-lag])
    return out
