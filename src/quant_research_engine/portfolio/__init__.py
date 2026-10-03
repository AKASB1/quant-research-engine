"""Portfolio-construction rules: weights from a cross-sectional signal (rows x N arrays; NaN =
not eligible). Weights are fractions of the sizing base.

- ``long_short_quantile(q, weighting, gross)``: with n eligible, k = floor(q n) longs (highest
  signal) and k shorts (lowest), ties by instrument order; ``equal`` weights or ``signal``
  weights proportional to |signal| within each leg; each leg holds gross / 2.
- ``long_only_topk(k)``: the k highest, equal weights summing to 1.
- ``vol_target(scale)``: weights proportional to signal / sigma (sigma: per-bar volatility per
  instrument), scaled so that the diagonal-risk estimate sqrt(sum w^2 sigma^2) equals ``scale``.
"""

from __future__ import annotations

import math

import numpy as np


def long_short_quantile(
    sig: np.ndarray, q: float = 0.2, weighting: str = "equal", gross: float = 1.0
) -> np.ndarray:
    s = np.atleast_2d(np.asarray(sig, dtype=np.float64))
    w = np.zeros(s.shape)
    for i in range(s.shape[0]):
        ok = np.flatnonzero(np.isfinite(s[i]))
        n = ok.size
        k = int(math.floor(q * n + 1e-9))
        if n < 2 or k < 1:
            continue
        order = ok[np.argsort(s[i, ok], kind="stable")]
        short, long_ = order[:k], order[n - k :]
        if weighting == "equal":
            w[i, long_] = 0.5 * gross / k
            w[i, short] = -0.5 * gross / k
        elif weighting == "signal":
            for leg, sgn in ((long_, 1.0), (short, -1.0)):
                a = np.abs(s[i, leg])
                tot = a.sum()
                w[i, leg] = (
                    sgn * 0.5 * gross * (a / tot if tot > 0 else np.full(leg.size, 1.0 / leg.size))
                )
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
    return w


def long_only_topk(sig: np.ndarray, k: int = 5) -> np.ndarray:
    s = np.atleast_2d(np.asarray(sig, dtype=np.float64))
    w = np.zeros(s.shape)
    for i in range(s.shape[0]):
        ok = np.flatnonzero(np.isfinite(s[i]))
        if ok.size == 0:
            continue
        kk = min(k, ok.size)
        order = ok[np.argsort(s[i, ok], kind="stable")]
        w[i, order[ok.size - kk :]] = 1.0 / kk
    return w


def vol_target(sig: np.ndarray, sigma: np.ndarray, scale: float = 0.01) -> np.ndarray:
    s = np.atleast_2d(np.asarray(sig, dtype=np.float64))
    v = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
    ok = np.isfinite(s) & np.isfinite(v) & (v > 0)
    raw = np.where(ok, s / np.where(ok, v, 1.0), 0.0)
    risk = np.sqrt((raw * raw * np.where(ok, v * v, 0.0)).sum(axis=1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(risk > 0, raw * (scale / risk), 0.0)


RULES = {
    "long_short_quantile": long_short_quantile,
    "long_only_topk": long_only_topk,
    "vol_target": vol_target,
}
