"""Cost model v1 of the shared contract (section 4): one function used by both engines.

For a fill of signed quantity ``q`` at reference price ``m`` with the decision-time estimates
``sigma`` (sigma_bar) and ``V`` (adv_shares):

    spread_cost = |q| m half_spread_bps / 1e4
    impact_bps  = 1e4 y sigma sqrt(|q| / V)  (sqrt) | 1e4 y sigma |q| / V  (linear) | 0  (none)
    impact_cost = |q| m impact_bps / 1e4
    price       = m (1 + sign(q) (half_spread_bps + impact_bps) / 1e4)
    commission  = max(min_commission, |q| m commission_bps / 1e4 + |q| commission_per_share)

Impact is 0 (and counted as ``impact_unavailable``) when sigma or V is unknown. Impact never
reads the fill bar's volume; the participation cap does (a physical limit, not a cost).
The functions work on NumPy scalars and arrays with the same arithmetic, so both engines get
bit-identical per-fill costs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ImpactCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["sqrt", "linear", "none"] = "sqrt"
    y: float = Field(default=0.5, ge=0)


class CostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    commission_bps: float = Field(default=1.0, ge=0)
    commission_per_share: float = Field(default=0.0, ge=0)
    min_commission: float = Field(default=0.0, ge=0)
    half_spread_bps: float = Field(default=2.0, ge=0)
    impact: ImpactCfg = Field(default_factory=ImpactCfg)
    borrow_bps_annual: float = Field(default=50.0, ge=0)
    financing_bps_annual: float = Field(default=100.0, ge=0)
    cash_rate_bps_annual: float = Field(default=0.0, ge=0)
    participation_cap: float = Field(default=0.1, gt=0)

    def scaled(self, mult: float) -> CostConfig:
        """Every cost rate multiplied by ``mult`` (cash rate and participation cap unchanged)."""
        return CostConfig(
            commission_bps=self.commission_bps * mult,
            commission_per_share=self.commission_per_share * mult,
            min_commission=self.min_commission * mult,
            half_spread_bps=self.half_spread_bps * mult,
            impact=ImpactCfg(model=self.impact.model, y=self.impact.y * mult),
            borrow_bps_annual=self.borrow_bps_annual * mult,
            financing_bps_annual=self.financing_bps_annual * mult,
            cash_rate_bps_annual=self.cash_rate_bps_annual,
            participation_cap=self.participation_cap,
        )


def zero_costs() -> CostConfig:
    return CostConfig().scaled(0.0)


def fill_costs(q, m, sigma, V, cfg: CostConfig):
    """(price, spread_cost, impact_cost, commission, impact_unavailable) for fills."""
    q = np.asarray(q, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    aq = np.abs(q)
    notional = aq * m
    hs = cfg.half_spread_bps
    unavailable = np.isnan(sigma) | np.isnan(V) | ~(V > 0)
    s_ = np.where(unavailable, 0.0, sigma)
    v_ = np.where(unavailable, 1.0, V)
    if cfg.impact.model == "sqrt":
        imp_bps = 1e4 * cfg.impact.y * s_ * np.sqrt(aq / v_)
    elif cfg.impact.model == "linear":
        imp_bps = 1e4 * cfg.impact.y * s_ * (aq / v_)
    else:
        imp_bps = np.zeros_like(aq)
        unavailable = np.zeros_like(unavailable)
    imp_bps = np.where(unavailable, 0.0, imp_bps)
    spread = notional * hs / 1e4
    impact = notional * imp_bps / 1e4
    price = m * (1.0 + np.sign(q) * (hs + imp_bps) / 1e4)
    comm = np.maximum(
        cfg.min_commission, notional * cfg.commission_bps / 1e4 + aq * cfg.commission_per_share
    )
    comm = np.where(q == 0, 0.0, comm)
    return price, spread, impact, comm, unavailable & (q != 0)


def accruals(cash_prev: float, short_value_prev: float, dt_days: float, cfg: CostConfig):
    """(borrow, financing, interest) accrued at a bar's close; ``short_value_prev`` is
    ``sum |q| P`` over short positions at the previous close."""
    borrow = short_value_prev * cfg.borrow_bps_annual / 1e4 * dt_days / 365.0
    financing = max(0.0, -cash_prev) * cfg.financing_bps_annual / 1e4 * dt_days / 365.0
    interest = max(0.0, cash_prev) * cfg.cash_rate_bps_annual / 1e4 * dt_days / 365.0
    return borrow, financing, interest
