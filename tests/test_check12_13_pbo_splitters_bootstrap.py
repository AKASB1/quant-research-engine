"""Check 12 (PBO, splitters, G6) and check 13 (stationary bootstrap)."""

import math

import numpy as np
import pytest

from quant_research_engine.inference.bootstrap import (
    bootstrap_stat,
    mean_rows,
    percentile_interval,
    stationary_indices,
)
from quant_research_engine.inference.pbo import pbo_cscv
from quant_research_engine.inference.splitters import (
    UnpurgedSplitError,
    cpcv,
    make_splits,
    purged_kfold,
    violations,
    walk_forward,
)
from quant_research_engine.rng import stream


def test_cscv_split_count():
    r = stream(1, "test.pbo").standard_normal((1003, 5))
    res = pbo_cscv(r)
    assert res.n_splits == math.comb(16, 8) == 12870
    assert res.rows_used == 1003 - 1003 % 16


def test_pbo_is_one_when_the_in_sample_best_is_always_the_out_of_sample_worst():
    """Trial B mirrors trial A (B = -A) and A has block means m_b = 1e-4 * (1 .. 15, -120) with
    alternating +-0.01 inside each block: A's total is zero and no 8-block subset sums to zero,
    so A's in-sample and out-of-sample means always have opposite signs."""
    S, n_per = 16, 20
    m = np.r_[np.arange(1, 16), -120.0] * 1e-4
    a = np.concatenate(
        [m[b] + 0.01 * np.where(np.arange(n_per) % 2 == 0, 1.0, -1.0) for b in range(S)]
    )
    res = pbo_cscv(np.column_stack([a, -a]))
    assert res.pbo == 1.0


def test_pbo_is_zero_when_one_trial_dominates_every_block_set():
    """Seed 5, stream test.pbo0: 20 trials of unit noise; trial 7 has drift 1.0 per bar."""
    r = stream(5, "test.pbo0").standard_normal((800, 20))
    r[:, 7] += 1.0
    assert pbo_cscv(r).pbo == 0.0


def test_pbo_does_not_depend_on_trial_order():
    r = stream(9, "test.pbo.perm").standard_normal((512, 12))
    perm = stream(9, "test.pbo.perm2").permutation(12)
    a, b = pbo_cscv(r), pbo_cscv(r[:, perm])
    assert a.pbo == b.pbo
    assert np.array_equal(np.sort(a.lambdas), np.sort(b.lambdas))


def test_purge_and_embargo_property_on_random_horizons():
    rng = stream(2, "test.purge")
    for _ in range(200):
        n = int(rng.integers(30, 400))
        k = int(rng.integers(2, 8))
        h = int(rng.integers(1, 25))
        folds = purged_kfold(n, k, h)
        assert violations(folds, h) == 0
        for train, test in folds:
            for i in train[:: max(1, len(train) // 20)].tolist():
                assert all(abs(i - j) >= h for j in test.tolist())
        for train, test in walk_forward(n, k, h):
            if len(train) and len(test):
                assert train.max() + h < test.min() + 1  # ends h bars before the first test obs
        assert violations(walk_forward(n, k, h), h) == 0


def test_g6_refuses_unpurged_splitters_on_overlapping_labels():
    rng = stream(3, "cv")
    for name in ("shuffled_kfold", "contiguous_kfold"):
        with pytest.raises(UnpurgedSplitError):
            make_splits(name, 300, 5, 5, rng=rng)
        s = make_splits(name, 300, 5, 5, rng=rng, allow_unpurged=True)
        assert s.label == "UNPURGED"
    assert make_splits("purged_kfold", 300, 5, 5).label == "purged"
    assert make_splits("walk_forward", 300, 5, 21).label == "purged"
    assert make_splits("contiguous_kfold", 300, 5, 1).label == "purged"  # h = 1: no overlap


def test_cpcv_counts():
    splits, paths = cpcv(600, 6, 2, 5)
    assert len(splits) == 15 and paths == 5
    for train, test, _ in splits:
        assert violations([(train, test)], 5) == 0


def test_bootstrap_is_deterministic_from_the_seed():
    a = stationary_indices(100, 100, 50, 0.1, stream(4, "bootstrap"))
    b = stationary_indices(100, 100, 50, 0.1, stream(4, "bootstrap"))
    c = stationary_indices(100, 100, 50, 0.1, stream(5, "bootstrap"))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("p", [0.01, 0.05, 0.1, 0.3])
def test_block_break_share_positions_2_to_1000(p):
    n = 1000
    idx = stationary_indices(n, 1000, 1000, p, stream(6, "bootstrap"))
    breaks = idx[:, 1:] != (idx[:, :-1] + 1) % n
    share = breaks.mean()
    expect = p * (1 - 1 / n)
    assert abs(share / expect - 1) <= 0.05, (share, expect)


@pytest.mark.slow
def test_percentile_interval_coverage_iid_and_ar1():
    """Coverage of the 95 % percentile interval of a mean, 300 replications, n = 250, B = 500.
    Tolerance stated in advance: iid with p = 1 (iid resampling) within [0.90, 0.98]; AR(1) with
    phi = 0.5 and mean block length 10 (p = 0.1) within [0.85, 0.98] (the percentile interval
    of a dependent-data mean under-covers somewhat in small samples)."""
    for phi, p, lo, hi in ((0.0, 1.0, 0.90, 0.98), (0.5, 0.1, 0.85, 0.98)):
        g = stream(7, f"test.cover.{phi}")
        hits = 0
        for _ in range(300):
            e = g.standard_normal(250 + 50)
            x = np.empty_like(e)
            x[0] = e[0]
            for t in range(1, len(e)):
                x[t] = phi * x[t - 1] + e[t]
            x = x[50:]
            vals = bootstrap_stat(x, mean_rows, 500, p, g)
            a, b = percentile_interval(vals)
            hits += a <= 0.0 <= b
        cov = hits / 300
        assert lo <= cov <= hi, (phi, cov)
