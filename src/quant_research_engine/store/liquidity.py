"""Liquidity inputs of the cost model, derived from bars by one pure function.

For the bar ``k`` of an instrument (row ``ts_event`` and ``ts_avail`` are the bar's), from the
bars ``<= k`` only:

- ``adv_shares``: mean volume of the last 20 bars, in the instrument's shares as of the bar:
  volumes of bars before a split's ex-bar (splits with ex-bar ``<= k``) are multiplied by the
  split's ratio, so a split inside the window does not halve or double the average;
- ``sigma_bar``: square root of the weighted mean of the squared returns of 2.5 over the last
  250 returns, weight ``0.5 ** (age / 20)`` with age 0 for the newest (returns are split- and
  dividend-neutral, so splits need no treatment here).

A row exists once 20 returns are available (from the 21st bar) and only when ``adv_shares`` is
positive. Sums run over lags in a fixed order (newest first), so the bulk computation and the
recomputation of one row from a store cut at the row's ``ts_avail`` agree bit for bit.
"""

from __future__ import annotations

import numpy as np

ADV_WINDOW = 20
SIGMA_WINDOW = 250
HALF_LIFE = 20.0
MIN_RETURNS = 20


def _weights() -> np.ndarray:
    return 0.5 ** (np.arange(SIGMA_WINDOW, dtype=np.float64) / HALF_LIFE)


def liquidity_rows(
    volume: np.ndarray, rets: np.ndarray, ratio: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(valid, adv_shares, sigma_bar) for every bar of one instrument (arrays in bar order).

    ``rets[j]`` is the return of 2.5 of bar j (NaN for the first bar); ``ratio[j]`` the split
    ratio effective in bar j (1 if none).
    """
    n = len(volume)
    adv = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return valid, adv, sig
    # Volume of bar k-lag in the shares of bar k: volume * C[k] / C[k-lag]; with the ratios of
    # the synthetic generator (2, 3) and of the tests the quotient of cumulative products is
    # exact. C is a running product in bar order.
    c = np.cumprod(ratio)
    acc_v = np.zeros(n)
    for lag in range(ADV_WINDOW):
        x = np.zeros(n)
        if lag < n:
            src = volume[: n - lag] * (c[lag:] / c[: n - lag])
            x[lag:] = src
        acc_v = acc_v + x
    w = _weights()
    r2 = np.where(np.isnan(rets), 0.0, rets * rets)
    has_r = ~np.isnan(rets)
    acc = np.zeros(n)
    wsum = np.zeros(n)
    count = np.zeros(n, dtype=np.int64)
    for lag in range(SIGMA_WINDOW):
        if lag >= n:
            break
        xr = np.zeros(n)
        xw = np.zeros(n)
        xc = np.zeros(n, dtype=np.int64)
        ok = has_r[: n - lag]
        xr[lag:] = np.where(ok, w[lag] * r2[: n - lag], 0.0)
        xw[lag:] = np.where(ok, w[lag], 0.0)
        xc[lag:] = ok
        acc = acc + xr
        wsum = wsum + xw
        count = count + xc
    idx = np.arange(n)
    valid = (count >= MIN_RETURNS) & (idx >= ADV_WINDOW - 1)
    adv[valid] = acc_v[valid] / ADV_WINDOW
    valid &= adv > 0
    sig[valid] = np.sqrt(acc[valid] / wsum[valid])
    adv[~valid] = np.nan
    sig[~valid] = np.nan
    return valid, adv, sig


def liquidity_row(
    volume: np.ndarray, rets: np.ndarray, ratio: np.ndarray
) -> tuple[bool, float, float]:
    """The row of the last bar of the given (cut) history; same arithmetic as the bulk path."""
    valid, adv, sig = liquidity_rows(volume, rets, ratio)
    if len(valid) == 0:
        return False, float("nan"), float("nan")
    return bool(valid[-1]), float(adv[-1]), float(sig[-1])
