"""Check 11: metrics on hand-made series and the known answers of the shared contract."""

import math

import numpy as np
import pytest

from quant_research_engine import metrics as M
from quant_research_engine.inference.sharpe import dsr, expected_max_sharpe, psr
from quant_research_engine.rng import stream

R = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02])


def rel(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


@pytest.mark.parametrize("ppy", [252, 365, 52, 12])
def test_hand_made_series(ppy):
    m = R.mean()
    sd = math.sqrt(sum((x - m) ** 2 for x in R) / 5)
    assert rel(M.sharpe(R), m / sd)
    assert rel(M.sharpe(R, ppy), m / sd * math.sqrt(ppy))
    dd = math.sqrt(sum(min(x, 0) ** 2 for x in R) / 6)
    assert rel(M.sortino(R, ppy), m / dd * math.sqrt(ppy))
    assert rel(M.ann_vol(R, ppy), sd * math.sqrt(ppy))
    eq = np.array([100.0, 110.0, 99.0, 120.0, 90.0, 95.0])
    assert rel(M.max_drawdown(eq), 30.0 / 120.0)
    assert rel(M.ann_return(eq, ppy), (95 / 100) ** (ppy / 5) - 1)
    assert rel(M.calmar(eq, ppy), ((95 / 100) ** (ppy / 5) - 1) / 0.25)


def test_cvar_hit_rate_skew_kurt():
    x = np.arange(1, 41, dtype=float) - 20.0  # -19 .. 20, T = 40
    assert M.cvar(x) == -(-19 - 18) / 2  # ceil(0.05 * 40) = 2 smallest
    r = np.array([0.01, -0.01, 0.02, 0.0])
    assert M.hit_rate(r, [1.0, 0.0, 2.0, 1.0]) == 2 / 3
    y = np.array([1.0, 2.0, 3.0, 10.0])
    m = y.mean()
    s = math.sqrt(np.mean((y - m) ** 2))
    assert rel(M.skew(y), np.mean((y - m) ** 3) / s**3)
    assert rel(M.kurt(y), np.mean((y - m) ** 4) / s**4)
    z = stream(0, "test.kurt").standard_normal(200_000)
    assert abs(M.kurt(z) - 3) < 0.05


def test_lo_known_answers():
    rho = [0.2**k for k in range(1, 300)]
    assert abs(M.lo_eta(12, rho=rho, L=10**6) - 2.8788486905) <= 1e-9 * 3
    assert abs(M.lo_eta(252, rho=rho, L=10**6) - 12.9722102136) <= 1e-9 * 13
    assert abs(M.lo_eta(12, rho=rho, L=10) - 2.8788486939) <= 1e-9 * 3
    assert abs(M.lo_eta(252, rho=rho, L=10) - 12.9722104255) <= 1e-9 * 13
    assert abs(M.lo_eta(12, rho=[0.0] * 20) - math.sqrt(12)) <= 1e-12


def test_psr_sr0_dsr_known_answers():
    sr = 1.25 / math.sqrt(252)
    assert abs(sr - 0.0787425985) <= 1e-9
    assert abs(psr(sr, 0.0, 250, -0.5, 6.0) - 0.8876753070) <= 1e-9
    assert abs(psr(sr, 0.5 / math.sqrt(252), 250, -0.5, 6.0) - 0.7668629015) <= 1e-9
    assert abs(expected_max_sharpe(10, 1.0) - 1.5745983013) <= 1e-9 * 1.6
    assert abs(expected_max_sharpe(100, 1.0) - 2.5306028932) <= 1e-9 * 2.6
    assert abs(expected_max_sharpe(1000, 1.0) - 3.2551215137) <= 1e-9 * 3.3
    v = 0.25 / 252
    assert abs(expected_max_sharpe(100, v) - 0.0797064991) <= 1e-9
    assert abs(expected_max_sharpe(100, v) * math.sqrt(252) - 1.2653014466) <= 1e-9 * 1.3
    assert abs(dsr(sr, 250, -0.5, 6.0, 100, v) - 0.4940703731) <= 1e-9
