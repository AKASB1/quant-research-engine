"""Point-in-time price adjustment and total returns (shared contract 2.5 and 3.2).

Raw prices are never adjusted by the whole corporate-action table. As of an instant ``t``:

- split-adjusted close: the raw close of every bar before the ex-bar of a split is divided by
  the split's ratio, for the splits with ``ts_ex <= t`` only (products taken in ascending
  ``ts_ex`` order, so the fast path and the brute-force check agree bit for bit);
- total-return series: the cumulative product of the returns of 2.5 of the bars known at ``t``,
  scaled so that its last value equals the last split-adjusted close. A dividend enters when
  its ex-bar is known. The shortcut ``prev_close * (1 - dividend / prev_close)`` is not used.
"""

from __future__ import annotations

import numpy as np


def ex_bar_index(bar_ts_open: np.ndarray, ts_ex: int) -> int:
    """Index of the first bar with ``ts_open >= ts_ex`` (len(bars) if none)."""
    return int(np.searchsorted(bar_ts_open, ts_ex, side="left"))


def bar_events(
    bar_ts_open: np.ndarray,
    actions: list[tuple[str, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar split ratio (1 if none) and dividend per pre-split share (0 if none).

    ``actions`` are (action, ts_ex, value) of one instrument; an action whose ex-bar is not in
    ``bar_ts_open`` has no effect here.
    """
    n = len(bar_ts_open)
    ratio = np.ones(n)
    div = np.zeros(n)
    for action, ts_ex, value in actions:
        j = ex_bar_index(bar_ts_open, ts_ex)
        if j >= n:
            continue
        if action == "split":
            ratio[j] = ratio[j] * value
        else:
            div[j] = div[j] + value
    return ratio, div


def bar_returns(close: np.ndarray, ratio: np.ndarray, div: np.ndarray) -> np.ndarray:
    """Returns of 2.5 for consecutive bars of one instrument; element 0 is NaN (first bar)."""
    r = np.full(len(close), np.nan)
    if len(close) > 1:
        r[1:] = (close[1:] * ratio[1:] + div[1:]) / close[:-1] - 1.0
    return r


def split_factor_asof(
    n_bars: int, bar_ts_open: np.ndarray, splits: list[tuple[int, float]], t: int
) -> np.ndarray:
    """Divisor per bar: product of the ratios of splits with ts_ex <= t whose ex-bar is later."""
    f = np.ones(n_bars)
    for ts_ex, ratio in sorted(splits):
        if ts_ex > t:
            continue
        j = ex_bar_index(bar_ts_open, ts_ex)
        jj = min(j, n_bars)
        if jj > 0:
            f[:jj] = f[:jj] * ratio
    return f


def split_adjusted_close_asof(
    close: np.ndarray, bar_ts_open: np.ndarray, splits: list[tuple[int, float]], t: int
) -> np.ndarray:
    """``close`` are the raw closes of the bars known at ``t`` (one instrument, in time order)."""
    return close / split_factor_asof(len(close), bar_ts_open, splits, t)


def split_adjusted_close_bruteforce(
    close: np.ndarray, bar_ts_open: np.ndarray, splits: list[tuple[int, float]], t: int
) -> np.ndarray:
    out = np.empty(len(close))
    for j in range(len(close)):
        f = 1.0
        for ts_ex, ratio in sorted(splits):
            if ts_ex <= t and bar_ts_open[j] < ts_ex:
                f = f * ratio
        out[j] = close[j] / f
    return out


def total_return_asof(adj_close: np.ndarray, rets: np.ndarray) -> np.ndarray:
    """Total-return series of the known bars: cumulative product of the returns, scaled so the
    last value equals the last split-adjusted close. ``rets[0]`` (first known bar) is ignored."""
    n = len(adj_close)
    if n == 0:
        return adj_close.copy()
    growth = np.ones(n)
    if n > 1:
        growth[1:] = np.cumprod(1.0 + rets[1:])
    return growth * (adj_close[-1] / growth[-1])
