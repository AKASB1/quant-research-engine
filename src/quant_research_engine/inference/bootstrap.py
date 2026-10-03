"""Stationary bootstrap (Politis and Romano) on the circle of T bars.

The first index of a draw is uniform; each later index is the successor (mod T) of the previous
one with probability 1 - p and a fresh uniform index with probability p; the mean block length
is 1/p. Block-length rule used here: fixed (``p`` given); intervals are percentile intervals.
Every draw comes from the stream the caller passes (``bootstrap``), so results are reproducible.
"""

from __future__ import annotations

import numpy as np


def stationary_indices(
    n: int, length: int, n_draws: int, p: float, rng: np.random.Generator
) -> np.ndarray:
    """(n_draws, length) index matrix."""
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    u = rng.random((n_draws, length))
    fresh = rng.integers(0, n, size=(n_draws, length))
    new = u < p
    new[:, 0] = True
    pos = np.arange(length)[None, :]
    start = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    base = np.take_along_axis(fresh, start, axis=1)
    return (base + (pos - start)) % n


def bootstrap_stat(x, stat, n_draws: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Statistic ``stat(sample) -> float`` (vectorized over rows) on stationary-bootstrap draws."""
    x = np.asarray(x, dtype=np.float64)
    idx = stationary_indices(len(x), len(x), n_draws, p, rng)
    return np.asarray(stat(x[idx]), dtype=np.float64)


def percentile_interval(values, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(values), [a, 1.0 - a])
    return float(lo), float(hi)


def sharpe_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, ddof=1)
    return np.where(sd > 0, x.mean(axis=1) / np.where(sd > 0, sd, 1.0), 0.0)


def mean_rows(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1)
