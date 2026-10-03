"""Probability of backtest overfitting by combinatorially symmetric cross-validation (CSCV).

Input: per-bar returns, T rows by N trials. The first ``T mod S`` rows are dropped and the rest
is cut into S equal blocks (default 16). For each of the C(S, S/2) choices of half the blocks as
in-sample: the in-sample best trial n* has the highest in-sample Sharpe ratio (ties: lowest
index); omega = (1 + number of trials with a strictly lower out-of-sample Sharpe) / (N + 1);
lambda = ln(omega / (1 - omega)). PBO is the share of splits with lambda <= 0. Block sums are
combined in a fixed block order, so the result does not depend on the order of the trials
except through the tie rule.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class PBOResult:
    pbo: float
    lambdas: np.ndarray
    n_splits: int
    n_trials: int
    rows_used: int


def _combos(S: int) -> np.ndarray:
    return np.asarray(
        [[1 if b in c else 0 for b in range(S)] for c in itertools.combinations(range(S), S // 2)],
        dtype=np.float64,
    )


def _sharpe_from_blocks(M: np.ndarray, s1: np.ndarray, s2: np.ndarray, n_per: int) -> np.ndarray:
    """Per-split Sharpe of each trial from block sums (fixed block order)."""
    tot1 = np.zeros((M.shape[0], s1.shape[1]))
    tot2 = np.zeros_like(tot1)
    for b in range(M.shape[1]):
        w = M[:, b : b + 1]
        tot1 = tot1 + w * s1[b]
        tot2 = tot2 + w * s2[b]
    n = M.sum(axis=1, keepdims=True) * n_per
    mean = tot1 / n
    var = (tot2 - n * mean * mean) / (n - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        # zero variance: Sharpe +-inf for a nonzero mean, 0 for a zero mean (never NaN)
        return np.where(
            sd > 0, mean / sd, np.where(mean > 0, np.inf, np.where(mean < 0, -np.inf, 0.0))
        )


def pbo_cscv(returns, S: int = 16) -> PBOResult:
    R = np.asarray(returns, dtype=np.float64)
    T, N = R.shape
    if S % 2 or S < 2:
        raise ValueError("S must be even")
    drop = T % S
    R = R[drop:]
    n_per = R.shape[0] // S
    blocks = R.reshape(S, n_per, N)
    s1 = blocks.sum(axis=1)
    s2 = (blocks * blocks).sum(axis=1)
    M = _combos(S)
    is_sr = _sharpe_from_blocks(M, s1, s2, n_per)
    oos_sr = _sharpe_from_blocks(1.0 - M, s1, s2, n_per)
    best = np.argmax(is_sr, axis=1)  # first maximum: lowest index on ties
    oos_best = oos_sr[np.arange(len(best)), best]
    lower = np.sum(oos_sr < oos_best[:, None], axis=1)
    omega = (1.0 + lower) / (N + 1.0)
    lam = np.log(omega / (1.0 - omega))
    return PBOResult(float(np.mean(lam <= 0)), lam, len(M), N, R.shape[0])
