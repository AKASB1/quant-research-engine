"""Declared, past-only features with two implementations that must agree.

A feature declares its inputs and lookback and has a context implementation
(``compute(ctx) -> (N,)`` for the universe at the decision instant) and a panel implementation
(``panel(src) -> (T, N)``, values at every bar, past-only by construction: windows that end at
the bar, sums over lags in a fixed order, no centered window, no negative shift). Both use the
same lag order, so they agree to rounding (checked to 1e-12) and the panel value at bar k is
unchanged when the data after k is removed (checked bit for bit). Built in:

- ``momentum(lookback, skip)``: prod over bars k-lookback+1 .. k-skip of (1 + r) - 1;
- ``reversal(lookback)``: minus the compounded return of the last ``lookback`` bars;
- ``low_vol(lookback)``: minus the standard deviation (ddof 1) of the last ``lookback`` returns;
- ``volume_trend(lookback)``: log of mean dollar volume (close x volume, split-invariant) of the
  last ``lookback`` bars over that of the ``lookback`` bars before;
- ``signal_x_ewma(halflife)``: exponentially weighted mean of the series ``<id>.signal_x`` over
  the last ``ceil(10 * halflife)`` bars, weights 0.5 ** (lag / halflife), normalized over the
  available values.

Returns are those of the shared contract (2.5); a window with a missing value gives NaN.
"""

from __future__ import annotations

import math

import numpy as np

from quant_research_engine.features.panel_source import PanelSource, derived_availability

__all__ = [
    "FEATURES",
    "Feature",
    "PanelSource",
    "derived_availability",
    "make_feature",
    "momentum",
    "reversal",
    "low_vol",
    "volume_trend",
    "signal_x_ewma",
]


def _lag(a: np.ndarray, lag: int) -> np.ndarray:
    """Row k holds a[k - lag] (NaN before the start)."""
    out = np.full(a.shape, np.nan)
    if lag == 0:
        return a.copy()
    if lag < a.shape[0]:
        out[lag:] = a[:-lag]
    return out


class Feature:
    name = "feature"
    inputs: tuple[str, ...] = ()
    series_suffixes: tuple[str, ...] = ()

    def __init__(self, **params):
        self.params = dict(params)

    @property
    def lookback(self) -> int:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return self.name + "(" + ",".join(f"{k}={v}" for k, v in sorted(self.params.items())) + ")"

    def compute(self, ctx) -> np.ndarray:
        raise NotImplementedError

    def panel(self, src: PanelSource) -> np.ndarray:
        raise NotImplementedError

    def declared_availability(self, src: PanelSource) -> np.ndarray:
        return src.declared_bar_avail()


class _ReturnProduct(Feature):
    """prod over lags in [lo, hi) of (1 + r[k - lag]) - 1, newest lag first."""

    inputs = ("returns",)

    def _lags(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def lookback(self) -> int:
        return self._lags()[1]

    def _sign(self) -> float:
        return 1.0

    def compute(self, ctx) -> np.ndarray:
        lo, hi = self._lags()
        r, _ = ctx.returns(hi)
        n = len(ctx.universe())
        if r.shape[0] < hi:
            return np.full(n, np.nan)
        acc = np.ones(n)
        for lag in range(lo, hi):
            acc = acc * (1.0 + r[r.shape[0] - 1 - lag])
        return self._sign() * (acc - 1.0)

    def panel(self, src: PanelSource) -> np.ndarray:
        lo, hi = self._lags()
        r = src.returns()
        acc = np.ones(r.shape)
        for lag in range(lo, hi):
            acc = acc * (1.0 + _lag(r, lag))
        return self._sign() * (acc - 1.0)


class momentum(_ReturnProduct):  # noqa: N801
    name = "momentum"

    def __init__(self, lookback: int = 21, skip: int = 5):
        if not 0 <= skip < lookback:
            raise ValueError("need 0 <= skip < lookback")
        super().__init__(lookback=lookback, skip=skip)

    def _lags(self):
        return self.params["skip"], self.params["lookback"]


class reversal(_ReturnProduct):  # noqa: N801
    name = "reversal"

    def __init__(self, lookback: int = 5):
        super().__init__(lookback=lookback)

    def _lags(self):
        return 0, self.params["lookback"]

    def _sign(self):
        return -1.0


class low_vol(Feature):  # noqa: N801
    name = "low_vol"
    inputs = ("returns",)

    def __init__(self, lookback: int = 63):
        super().__init__(lookback=lookback)

    @property
    def lookback(self):
        return self.params["lookback"]

    @staticmethod
    def _std(rows: list[np.ndarray]) -> np.ndarray:
        L = len(rows)
        s = np.zeros(rows[0].shape)
        for x in rows:
            s = s + x
        m = s / L
        q = np.zeros(rows[0].shape)
        for x in rows:
            q = q + (x - m) * (x - m)
        return -np.sqrt(q / (L - 1))

    def compute(self, ctx):
        L = self.lookback
        r, _ = ctx.returns(L)
        if r.shape[0] < L:
            return np.full(len(ctx.universe()), np.nan)
        return self._std([r[r.shape[0] - 1 - lag] for lag in range(L)])

    def panel(self, src):
        r = src.returns()
        return self._std([_lag(r, lag) for lag in range(self.lookback)])


class volume_trend(Feature):  # noqa: N801
    name = "volume_trend"
    inputs = ("bars.close", "bars.volume")

    def __init__(self, lookback: int = 21):
        super().__init__(lookback=lookback)

    @property
    def lookback(self):
        return 2 * self.params["lookback"] - 1

    def _calc(self, dv_rows: list[np.ndarray]) -> np.ndarray:
        L = self.params["lookback"]
        a = np.zeros(dv_rows[0].shape)
        b = np.zeros(dv_rows[0].shape)
        for lag in range(L):
            a = a + dv_rows[lag]
        for lag in range(L, 2 * L):
            b = b + dv_rows[lag]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log((a / L) / (b / L))
        return np.where((a > 0) & (b > 0), out, np.nan)

    def compute(self, ctx):
        n2 = 2 * self.params["lookback"]
        c, _ = ctx.bars("close", n2)
        v, _ = ctx.bars("volume", n2)
        if c.shape[0] < n2:
            return np.full(len(ctx.universe()), np.nan)
        dv = c * v
        return self._calc([dv[dv.shape[0] - 1 - lag] for lag in range(n2)])

    def panel(self, src):
        dv = src.bars("close") * src.bars("volume")
        return self._calc([_lag(dv, lag) for lag in range(2 * self.params["lookback"])])


class signal_x_ewma(Feature):  # noqa: N801
    name = "signal_x_ewma"
    inputs = ("series:.signal_x",)
    series_suffixes = (".signal_x",)

    def __init__(self, halflife: float = 3):
        super().__init__(halflife=halflife)

    @property
    def window(self) -> int:
        return int(math.ceil(10 * self.params["halflife"]))

    @property
    def lookback(self):
        return self.window - 1

    def _calc(self, rows: list[np.ndarray]) -> np.ndarray:
        h = float(self.params["halflife"])
        num = np.zeros(rows[0].shape)
        den = np.zeros(rows[0].shape)
        for lag, x in enumerate(rows):
            w = 0.5 ** (lag / h)
            ok = ~np.isnan(x)
            num = num + np.where(ok, w * np.where(ok, x, 0.0), 0.0)
            den = den + np.where(ok, w, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def compute(self, ctx):
        W = self.window
        _, te = ctx.bars("close", W)
        ids = ctx.universe()
        x = np.full((len(te), len(ids)), np.nan)
        for j, iid in enumerate(ids):
            try:
                ste, sval = ctx.series(f"{iid}.signal_x")
            except KeyError:
                continue
            # latest value with ts_event at or before each session (forward fill, as the panel
            # path does; identical to it when each value is released at its own bar's close)
            pos = np.searchsorted(ste, te, side="right") - 1
            ok = pos >= 0
            x[ok, j] = sval[pos[ok]]
        rows = [
            x[x.shape[0] - 1 - lag] if lag < x.shape[0] else np.full(len(ids), np.nan)
            for lag in range(W)
        ]
        return self._calc(rows)

    def panel(self, src):
        x = src.series_asof(".signal_x")
        return self._calc([_lag(x, lag) for lag in range(self.window)])


FEATURES = {
    "momentum": momentum,
    "reversal": reversal,
    "low_vol": low_vol,
    "volume_trend": volume_trend,
    "signal_x_ewma": signal_x_ewma,
}


def make_feature(name: str, **params) -> Feature:
    return FEATURES[name](**params)
