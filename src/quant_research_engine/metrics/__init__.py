"""Performance metrics of the shared contract (section 6). Inputs are per-bar net returns ``r``
(and equity including the opening value ``E_0`` where stated); ``ppy`` is declared, never
inferred. Conventions: standard deviations with ddof = 1, skewness and raw kurtosis with the
1/T normalization (a normal distribution has kurtosis 3)."""

from __future__ import annotations

import math

import numpy as np


def _r(r) -> np.ndarray:
    return np.asarray(r, dtype=np.float64)


def ann_return(equity, ppy: int) -> float:
    """(E_T / E_0) ** (ppy / T) - 1 with T the number of bars after E_0."""
    e = _r(equity)
    T = len(e) - 1
    return float((e[-1] / e[0]) ** (ppy / T) - 1.0)


def ann_vol(r, ppy: int) -> float:
    return float(np.std(_r(r), ddof=1) * math.sqrt(ppy))


def sharpe(r, ppy: int | None = None, rf: float = 0.0) -> float:
    """mean(r - rf) / std(r - rf, ddof=1), times sqrt(ppy) when ppy is given (per bar if not)."""
    x = _r(r) - rf
    sd = np.std(x, ddof=1)
    s = float(np.mean(x) / sd) if sd > 0 else math.nan
    return s * math.sqrt(ppy) if ppy else s


def sortino(r, ppy: int | None = None, rf: float = 0.0) -> float:
    """mean(r - rf) / sqrt(mean(min(r - rf, 0)^2)) over all T bars, times sqrt(ppy)."""
    x = _r(r) - rf
    dd = math.sqrt(float(np.mean(np.minimum(x, 0.0) ** 2)))
    s = float(np.mean(x)) / dd if dd > 0 else math.nan
    return s * math.sqrt(ppy) if ppy else s


def max_drawdown(equity) -> float:
    """Largest (peak - E) / peak with the running peak including E_0."""
    e = _r(equity)
    peak = np.maximum.accumulate(e)
    return float(np.max((peak - e) / peak))


def calmar(equity, ppy: int) -> float:
    mdd = max_drawdown(equity)
    return ann_return(equity, ppy) / mdd if mdd > 0 else math.nan


def cvar(r, level: float = 0.95) -> float:
    """Minus the mean of the ceil((1 - level) T) smallest returns (a positive loss)."""
    x = np.sort(_r(r))
    n = max(1, math.ceil((1.0 - level) * len(x) - 1e-12))
    return float(-np.mean(x[:n]))


def hit_rate(r, exposure_prev) -> float:
    """Share of bars with r > 0 among bars with nonzero exposure at the previous close."""
    x = _r(r)
    on = _r(exposure_prev) != 0
    return float(np.mean(x[on] > 0)) if on.any() else math.nan


def skew(r) -> float:
    x = _r(r)
    m = x.mean()
    s = math.sqrt(float(np.mean((x - m) ** 2)))
    return float(np.mean((x - m) ** 3) / s**3) if s > 0 else math.nan


def kurt(r) -> float:
    """Raw (not excess) kurtosis, 1/T normalization."""
    x = _r(r)
    m = x.mean()
    s2 = float(np.mean((x - m) ** 2))
    return float(np.mean((x - m) ** 4) / s2**2) if s2 > 0 else math.nan


def autocorr(r, max_lag: int) -> np.ndarray:
    """Sample autocorrelations rho_1..rho_max_lag (centered, denominator sum of squares)."""
    x = _r(r)
    x = x - x.mean()
    den = float(np.dot(x, x))
    out = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        out[k - 1] = float(np.dot(x[k:], x[:-k])) / den if k < len(x) and den > 0 else 0.0
    return out


def lo_eta(q: int, rho=None, r=None, L: int = 10) -> float:
    """eta(q) = q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho_k), with rho_k for k up to min(q-1, L)
    and 0 beyond. ``rho`` (rho_1, rho_2, ...) is given, or estimated from returns ``r``."""
    kmax = min(q - 1, L)
    if rho is None:
        rho = autocorr(r, kmax) if kmax > 0 else np.zeros(0)
    rho = np.asarray(rho, dtype=np.float64)
    s = 0.0
    for k in range(1, kmax + 1):
        s += (q - k) * float(rho[k - 1])
    return q / math.sqrt(q + 2.0 * s)


def lo_sharpe(r, ppy: int, L: int = 10) -> float:
    """Autocorrelation-adjusted annualized Sharpe (Lo, 2002): eta(ppy) times the per-bar SR."""
    return lo_eta(ppy, r=r, L=L) * sharpe(r)


def summary(r, equity, ppy: int, exposure_prev=None) -> dict:
    out = {
        "ann_return": ann_return(equity, ppy),
        "ann_vol": ann_vol(r, ppy),
        "sharpe": sharpe(r, ppy),
        "sr_bar": sharpe(r),
        "sortino": sortino(r, ppy),
        "max_drawdown": max_drawdown(equity),
        "cvar_95": cvar(r),
        "skew": skew(r),
        "kurt": kurt(r),
    }
    out["calmar"] = out["ann_return"] / out["max_drawdown"] if out["max_drawdown"] > 0 else math.nan
    if exposure_prev is not None:
        out["hit_rate"] = hit_rate(r, exposure_prev)
    return out
