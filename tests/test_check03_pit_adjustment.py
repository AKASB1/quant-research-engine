"""Check 3 (point-in-time part): split-adjusted and total-return series as of an instant."""

import numpy as np
from helpers import T, golden_store

from quant_research_engine.rng import stream
from quant_research_engine.store.adjust import (
    bar_events,
    bar_returns,
    split_adjusted_close_asof,
    split_adjusted_close_bruteforce,
    total_return_asof,
)

DAY = 86_400_000_000


def _random_instrument(rng, n):
    to = (np.arange(n, dtype=np.int64) * DAY) + 10 * DAY
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    actions = []
    for j in rng.choice(np.arange(1, n), size=min(n - 1, int(rng.integers(0, 5))), replace=False):
        if rng.random() < 0.5:
            actions.append(
                (
                    "split",
                    int(to[j] - int(rng.integers(0, 2)) * DAY // 2),
                    float(rng.choice([0.5, 1.5, 2.0, 3.0])),
                )
            )
        else:
            actions.append(("cash_dividend", int(to[j]), float(round(rng.uniform(0.05, 1.0), 2))))
    return to, close, actions


def test_split_adjusted_equals_bruteforce_and_differs_across_instants():
    rng = stream(21, "test.pit")
    for _ in range(300):
        n = int(rng.integers(2, 40))
        to, close, acts = _random_instrument(rng, n)
        splits = [(ex, v) for a, ex, v in acts if a == "split"]
        for t in rng.integers(0, n + 15, size=5).tolist():
            ti = int(t) * DAY
            a = split_adjusted_close_asof(close, to, splits, ti)
            b = split_adjusted_close_bruteforce(close, to, splits, ti)
            assert a.tolist() == b.tolist()  # exact
        # a split between two instants changes the adjustment of earlier bars
        for ex, r in splits:
            j = int(np.searchsorted(to, ex))
            if j == 0 or j >= n or r == 1.0:
                continue
            before = split_adjusted_close_asof(close, to, splits, ex - 1)
            after = split_adjusted_close_asof(close, to, splits, ex)
            assert not np.array_equal(before[:j], after[:j])


def test_total_return_matches_returns_including_shortcut_breaking_cases():
    rng = stream(22, "test.tr")
    for _ in range(300):
        n = int(rng.integers(3, 40))
        to, close, acts = _random_instrument(rng, n)
        splits = [(ex, v) for a, ex, v in acts if a == "split"]
        t = int(to[-1]) + DAY
        ratio, div = bar_events(to, acts)
        rets = bar_returns(close, ratio, div)
        adj = split_adjusted_close_asof(close, to, splits, t)
        tr = total_return_asof(adj, rets)
        growth = np.cumprod(np.r_[1.0, 1.0 + rets[1:]])
        expect = growth * (adj[-1] / growth[-1])
        assert np.allclose(tr, expect, rtol=1e-12, atol=0)
        assert abs(tr[-1] - adj[-1]) <= 1e-12 * abs(adj[-1])
        tr_rets = tr[1:] / tr[:-1] - 1
        assert np.allclose(tr_rets, rets[1:], rtol=0, atol=1e-12)


def test_appendix_a_shortcut_counterexample():
    # previous close 102.0, dividend 0.5, ex-bar close 101.7
    r = bar_returns(np.array([102.0, 101.7]), np.array([1.0, 1.0]), np.array([0.0, 0.5]))
    assert abs(r[1] - 0.0019607843137254832) <= 1e-15
    shortcut = 101.7 / (102.0 - 0.5) - 1
    assert abs(shortcut - 0.0019704433497538254) <= 1e-15
    assert abs(r[1] - shortcut) > 1e-6
    tr = total_return_asof(np.array([102.0, 101.7]), r)
    assert abs(tr[1] / tr[0] - 1 - r[1]) <= 1e-12


def test_golden_point_in_time_answers():
    s = golden_store()
    out = s.adjusted_close(["B"], 10, knowledge=T("2024-03-06T21:00:00Z"))["B"][1]
    assert np.allclose(out, [20.2, 20.15, 20.0], rtol=1e-12)
    out = s.adjusted_close(["B"], 10, knowledge=T("2024-03-04T21:00:00Z"))["B"][1]
    assert out.tolist() == [40.4]
    tr = s.adjusted_close(["A"], 10, knowledge=T("2024-03-06T21:00:00Z"), total_return=True)["A"][1]
    assert np.allclose(tr, [100.50490196078431, 101.5, 101.5], rtol=1e-12, atol=0)
    tr = s.adjusted_close(["A"], 10, knowledge=T("2024-03-05T21:00:00Z"), total_return=True)["A"][1]
    assert np.allclose(tr, [101.0, 102.0], rtol=1e-12, atol=0)
    p = s.panel()
    b = p.ids.index("B")
    a = p.ids.index("A")
    assert abs(p.ret[1, b] - ((20.15 * 2.0) / 40.4 - 1)) <= 1e-15
    assert p.ret[2, a] == (101.5 + 0.5) / 102.0 - 1
