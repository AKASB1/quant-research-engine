"""Probabilistic and deflated Sharpe ratios (Bailey and Lopez de Prado), per-bar quantities.

    PSR(SR*) = Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1) / 4 SR^2))
    SR0      = sqrt(V) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e))),  g = Euler-Mascheroni
    DSR      = PSR(SR0)

``kurt`` is the raw kurtosis; ``V`` the variance of the per-bar Sharpe ratios of the N trials.
"""

from __future__ import annotations

import math

from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def psr(sr: float, sr_star: float, t_bars: int, skew: float, kurt: float) -> float:
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if den <= 0:
        return math.nan
    return float(norm.cdf((sr - sr_star) * math.sqrt(t_bars - 1) / math.sqrt(den)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """SR0: expected maximum per-bar Sharpe of N independent trials with Sharpe variance V."""
    if n_trials < 2:
        return 0.0
    g = EULER_GAMMA
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1.0 - g) * a + g * b))


def dsr(sr: float, t_bars: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> float:
    return psr(sr, expected_max_sharpe(n_trials, var_sr), t_bars, skew, kurt)
