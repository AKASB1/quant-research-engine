"""Validation splitters and the cross-validation guard (G6).

Observation ``i`` has a label that looks ``h`` bars ahead (it uses bars i+1 .. i+h).

- ``purged_kfold``: K contiguous test blocks; for a test block [a, b) every training observation
  i with a - h <= i < b + e is removed (embargo e, default h). Property: for every training i
  and test j the label windows (i, i+h] and (j, j+h] are disjoint.
- ``walk_forward``: K+1 contiguous blocks; fold f tests block f and trains on observations that
  end at least h bars before the block's first observation.
- ``contiguous_kfold`` (no purge) and ``shuffled_kfold``: the mistakes, kept to measure them.
- ``cpcv``: combinatorial purged cross-validation with N groups and k test groups:
  C(N, k) splits and C(N, k) k / N backtest paths.

G6: :func:`make_splits` refuses a splitter that violates the purge property for the label
horizon in use unless ``allow_unpurged`` is given; such splits are labelled ``UNPURGED``.
"""

from __future__ import annotations

import itertools
import math
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


class UnpurgedSplitError(ValueError):
    """A splitter would let training labels overlap test labels (guard G6)."""


_observer: list | None = None


@contextmanager
def observe():
    """Within the block, every split set created with violations (allowed or not) is recorded,
    so the audit sees an unpurged validation that a subject did not declare (guard G6)."""
    global _observer
    prev = _observer
    _observer = []
    try:
        yield _observer
    finally:
        _observer = prev


@dataclass
class Splits:
    name: str
    folds: list[tuple[np.ndarray, np.ndarray]]  # (train, test) observation indices
    horizon: int
    label: str  # "purged" or "UNPURGED"


def _blocks(n: int, k: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]


def purged_kfold(n: int, k: int, h: int, embargo: int | None = None):
    e = h if embargo is None else embargo
    out = []
    idx = np.arange(n)
    for a, b in _blocks(n, k):
        test = idx[a:b]
        train = idx[(idx < a - h) | (idx >= b + e)]
        out.append((train, test))
    return out


def walk_forward(n: int, k: int, h: int):
    out = []
    idx = np.arange(n)
    for a, b in _blocks(n, k + 1)[1:]:
        out.append((idx[: max(0, a - h)], idx[a:b]))
    return out


def contiguous_kfold(n: int, k: int):
    idx = np.arange(n)
    return [(idx[(idx < a) | (idx >= b)], idx[a:b]) for a, b in _blocks(n, k)]


def shuffled_kfold(n: int, k: int, rng: np.random.Generator):
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for f in range(k):
        test = np.sort(folds[f])
        train = np.sort(np.concatenate([folds[g] for g in range(k) if g != f]))
        out.append((train, test))
    return out


def cpcv(n: int, n_groups: int, k_test: int, h: int, embargo: int | None = None):
    """Splits of combinatorial purged CV and the number of backtest paths C(N,k) k / N."""
    e = h if embargo is None else embargo
    groups = _blocks(n, n_groups)
    idx = np.arange(n)
    out = []
    for combo in itertools.combinations(range(n_groups), k_test):
        test_mask = np.zeros(n, dtype=bool)
        purge = np.zeros(n, dtype=bool)
        for g in combo:
            a, b = groups[g]
            test_mask[a:b] = True
            purge |= (idx >= a - h) & (idx < b + e)
        out.append((idx[~purge & ~test_mask], idx[test_mask], combo))
    n_paths = math.comb(n_groups, k_test) * k_test // n_groups
    return out, n_paths


def violations(folds, h: int) -> int:
    """Number of (fold, test obs) whose label window overlaps a training label window."""
    bad = 0
    for train, test in folds:
        if len(train) == 0 or len(test) == 0:
            continue
        tr = np.sort(train)
        for j in test.tolist():
            # windows (i, i+h] and (j, j+h] overlap iff |i - j| < h
            lo = np.searchsorted(tr, j - h + 1, side="left")
            hi = np.searchsorted(tr, j + h - 1, side="right")
            if hi > lo:
                bad += 1
    return bad


def make_splits(
    name: str,
    n: int,
    k: int,
    h: int,
    *,
    allow_unpurged: bool = False,
    rng: np.random.Generator | None = None,
    embargo: int | None = None,
) -> Splits:
    if name == "purged_kfold":
        folds = purged_kfold(n, k, h, embargo)
    elif name == "walk_forward":
        folds = walk_forward(n, k, h)
    elif name == "contiguous_kfold":
        folds = contiguous_kfold(n, k)
    elif name == "shuffled_kfold":
        if rng is None:
            raise ValueError("shuffled_kfold needs a random stream (cv)")
        folds = shuffled_kfold(n, k, rng)
    else:
        raise ValueError(f"unknown splitter {name!r}")
    bad = violations(folds, h)
    if bad and _observer is not None:
        _observer.append((name, n, k, h, bad))
    if bad and not allow_unpurged:
        raise UnpurgedSplitError(
            f"{name} lets training labels overlap test labels for horizon {h} ({bad} test "
            "observations); use purged_kfold or walk_forward, or pass allow_unpurged"
        )
    return Splits(name, folds, h, "UNPURGED" if bad else "purged")
