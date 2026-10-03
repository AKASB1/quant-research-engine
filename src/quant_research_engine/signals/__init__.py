"""Cross-sectional transforms over the instruments of the universe at the decision instant.

Inputs are (rows, N) arrays (a context passes one row); NaN marks an instrument outside the
universe or without a value, and stays NaN. Ties keep instrument-id order.
"""

from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    """(x - mean) / std (ddof 1) per row over the finite entries; NaN with fewer than 2."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    ok = np.isfinite(x)
    n = ok.sum(axis=1, keepdims=True)
    xs = np.where(ok, x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = xs.sum(axis=1, keepdims=True) / n
        d = np.where(ok, x - m, 0.0)
        sd = np.sqrt((d * d).sum(axis=1, keepdims=True) / (n - 1))
        z = np.where(ok & (n >= 2) & (sd > 0), d / sd, np.nan)
    return z


def rank(x: np.ndarray) -> np.ndarray:
    """Ranks 0 .. n-1 per row among finite entries (ties by instrument order), scaled to [0, 1]."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = np.full(x.shape, np.nan)
    for i in range(x.shape[0]):
        ok = np.flatnonzero(np.isfinite(x[i]))
        if ok.size < 2:
            continue
        order = ok[np.argsort(x[i, ok], kind="stable")]
        out[i, order] = np.arange(ok.size) / (ok.size - 1)
    return out


def winsorize(x: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = x.copy()
    for i in range(x.shape[0]):
        ok = np.isfinite(x[i])
        if ok.sum() < 2:
            continue
        a, b = np.quantile(x[i, ok], [lo, hi])
        out[i, ok] = np.clip(x[i, ok], a, b)
    return out


TRANSFORMS = {
    "zscore": zscore,
    "rank": rank,
    "winsorize": winsorize,
    "none": lambda x: np.atleast_2d(x),
}
