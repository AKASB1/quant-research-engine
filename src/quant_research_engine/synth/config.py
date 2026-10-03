"""Configuration of the synthetic market (pydantic models; every number is an assumed value).

The meaning of each parameter follows the shared contract, section 7; ``docs/data.md`` lists
the defaults and the committed configurations under ``configs/synth/``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactorCfg(_Model):
    name: str
    annual_vol: float = Field(ge=0)
    mean_annual: float = 0.0


def _default_factors() -> list[FactorCfg]:
    return [
        FactorCfg(name="market", annual_vol=0.16),
        FactorCfg(name="style1", annual_vol=0.08),
        FactorCfg(name="style2", annual_vol=0.06),
    ]


class RegimeCfg(_Model):
    enabled: bool = True
    stress_vol_mult: float = Field(default=2.0, gt=0)
    p_calm_to_stress: float = Field(default=0.01, ge=0, le=1)
    p_stress_to_calm: float = Field(default=0.05, ge=0, le=1)


class VolumeCfg(_Model):
    adv_median: float = Field(default=1e6, gt=0)
    adv_log_sd: float = Field(default=1.0, ge=0)
    ar_phi: float = Field(default=0.8, ge=0, lt=1)
    ar_sd: float = Field(default=0.3, ge=0)
    zero_volume_prob: float = Field(default=0.001, ge=0, le=1)


class EventsCfg(_Model):
    enabled: bool = True
    delist_hazard_annual: float = Field(default=0.04, ge=0)
    delist_return_mean: float = -0.3
    delist_return_sd: float = Field(default=0.25, ge=0)
    split_rate_annual: float = Field(default=0.08, ge=0)
    split_ratios: list[float] = Field(default_factory=lambda: [2.0, 3.0])
    dividend_yield_annual: float = Field(default=0.02, ge=0)
    dividend_share: float = Field(default=0.5, ge=0, le=1)
    dividend_every_bars: int = Field(default=63, ge=1)
    announce_lead_bars: int = Field(default=10, ge=0)
    late_listing_share: float = Field(default=0.1, ge=0, le=1)
    min_life_bars: int = Field(default=30, ge=1)


class ShiftCfg(_Model):
    at_bar: int = Field(ge=1)
    vol_mult: float = Field(default=1.0, gt=0)
    corr_mult: float = Field(default=1.0, gt=0)
    ic_mult: float = 1.0
    mu_shift_annual: float = 0.0


class FundXCfg(_Model):
    period_bars: int = Field(default=21, ge=2)
    lag0_bars: int = Field(default=5, ge=1)
    lag1_bars: int = Field(default=15, ge=2)
    w: float = 2.0


class SynthConfig(_Model):
    name: str
    n_instruments: int = Field(ge=1)
    n_bars: int = Field(ge=2)
    calendar: Literal["equity_daily", "crypto_daily"] = "equity_daily"
    start: str = "2010-01-04"
    ppy: int = Field(default=252, ge=1)
    factors: list[FactorCfg] = Field(default_factory=_default_factors)
    beta_market_mean: float = 1.0
    beta_market_sd: float = Field(default=0.25, ge=0)
    beta_style_sd: float = Field(default=0.5, ge=0)
    idio_vol_median: float = Field(default=0.30, gt=0)
    idio_vol_log_sd: float = Field(default=0.25, ge=0)
    premium_annual: float = 0.0
    regime: RegimeCfg = Field(default_factory=RegimeCfg)
    tails: Literal["student_t", "gaussian"] = "student_t"
    gap_share: float = Field(default=0.25, ge=0, le=1)
    ic: float = Field(default=0.02, ge=-1, le=1)
    persistence: float = Field(default=0.9, ge=0, lt=1)
    observable_noise: float = Field(default=1.0, ge=0)
    volume: VolumeCfg = Field(default_factory=VolumeCfg)
    price_median: float = Field(default=40.0, gt=0)
    price_log_sd: float = Field(default=0.6, ge=0)
    range_scale: float = Field(default=0.5, ge=0)
    events: EventsCfg = Field(default_factory=EventsCfg)
    shift: ShiftCfg | None = None
    fund_x: FundXCfg = Field(default_factory=FundXCfg)
    lot_size: float = Field(default=0.0, ge=0)
    tick_size: float = Field(default=0.01, ge=0)

    def with_overrides(self, **kw) -> SynthConfig:
        data = self.model_dump()
        for k, v in kw.items():
            if isinstance(v, dict) and isinstance(data.get(k), dict):
                data[k] = {**data[k], **v}
            else:
                data[k] = v
        return SynthConfig.model_validate(data)


def load_synth_config(path: str) -> SynthConfig:
    from quant_research_engine.data.manifest import read_json

    return SynthConfig.model_validate(read_json(path))
