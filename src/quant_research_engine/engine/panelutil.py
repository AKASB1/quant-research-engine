"""Panel helpers shared by the engines: instrument bar numbers and as-of liquidity."""

from __future__ import annotations

import numpy as np


def bar_numbers(has_bar: np.ndarray) -> np.ndarray:
    """(T, N) index of each instrument's bar among its own bars (valid where has_bar)."""
    return np.cumsum(has_bar, axis=0) - 1


def last_bar_at(has_bar: np.ndarray) -> np.ndarray:
    """(T, N) instrument bar index of the last bar at or before each session (-1 if none)."""
    return np.cumsum(has_bar, axis=0) - 1


def liquidity_asof_panel(p) -> tuple[np.ndarray, np.ndarray]:
    """(adv, sigma) known at the ts_event of every session: the latest liquidity row with
    ``ts_avail <= ts_event[k]`` at or before session k (NaN if none)."""
    T, N = p.shape
    if T == 0:
        return np.zeros((0, N)), np.zeros((0, N))
    idx = np.where(~np.isnan(p.adv), np.arange(T)[:, None], -1)
    fwd = np.maximum.accumulate(idx, axis=0)
    t = p.ts_event[:, None]
    r = fwd.copy()
    cols = np.broadcast_to(np.arange(N), (T, N))
    for _ in range(T + 1):
        ok = r >= 0
        bad = ok & (p.liq_avail[np.maximum(r, 0), cols] > t)
        if not bad.any():
            break
        prev = np.where(r > 0, fwd[np.maximum(r - 1, 0), cols], -1)
        r = np.where(bad, prev, r)
    adv = np.where(r >= 0, p.adv[np.maximum(r, 0), cols], np.nan)
    sig = np.where(r >= 0, p.sigma[np.maximum(r, 0), cols], np.nan)
    return adv, sig
