"""Composition of one feature, one cross-sectional transform, and one portfolio rule.

``decide(ctx)`` is the context path (what the audit replays); ``weights_panel(src, rows)`` is the
panel path the vectorized experiments use, with the same transform and rule implementations
(the tests check that the two paths give the same weights).
"""

from __future__ import annotations

import numpy as np

from quant_research_engine.context import Strategy, TargetWeights, weights_from_array
from quant_research_engine.features import low_vol, make_feature
from quant_research_engine.portfolio import RULES
from quant_research_engine.signals import TRANSFORMS


class FeatureRuleStrategy(Strategy):
    name = "feature_rule"

    def __init__(
        self,
        feature: str = "momentum",
        feature_params: dict | None = None,
        transform: str = "zscore",
        rule: str = "long_short_quantile",
        rule_params: dict | None = None,
        vol_lookback: int = 63,
    ):
        super().__init__(
            feature=feature,
            feature_params=feature_params or {},
            transform=transform,
            rule=rule,
            rule_params=rule_params or {},
            vol_lookback=vol_lookback,
        )
        self.feature = make_feature(feature, **(feature_params or {}))
        self.transform = transform
        self.rule = rule
        self.rule_params = dict(rule_params or {})
        self.vol_lookback = vol_lookback
        extra = vol_lookback if rule == "vol_target" else 0
        self.history_bars = max(self.feature.lookback, extra) + 2
        self.series_suffixes = self.feature.series_suffixes

    def _apply(self, x: np.ndarray, sigma: np.ndarray | None) -> np.ndarray:
        z = TRANSFORMS[self.transform](x)
        if self.rule == "vol_target":
            return RULES[self.rule](z, sigma, **self.rule_params)
        return RULES[self.rule](z, **self.rule_params)

    def decide(self, ctx):
        ids = ctx.universe()
        if not ids:
            return TargetWeights({})
        x = self.feature.compute(ctx)
        sigma = None
        if self.rule == "vol_target":
            sigma = -low_vol(self.vol_lookback).compute(ctx)
        w = self._apply(x[None, :], None if sigma is None else sigma[None, :])[0]
        return weights_from_array(ids, w)

    def weights_panel(
        self, src, rows: np.ndarray, feature_panel: np.ndarray | None = None
    ) -> np.ndarray:
        """Weights at calendar rows ``rows`` for all instruments of the store (0 outside the
        universe). ``feature_panel`` may be passed to reuse a computed feature."""
        f = self.feature.panel(src) if feature_panel is None else feature_panel
        uni = src.universe_mask()
        x = np.where(uni[rows], f[rows], np.nan)
        sigma = None
        if self.rule == "vol_target":
            sigma = np.where(uni[rows], -low_vol(self.vol_lookback).panel(src)[rows], np.nan)
        return self._apply(x, sigma)


class FeatureOnly(FeatureRuleStrategy):
    """A built-in feature audited as a one-feature strategy: zscore, weights z / sum |z|."""

    name = "feature_only"

    def __init__(self, feature: str = "momentum", feature_params: dict | None = None):
        super().__init__(
            feature=feature,
            feature_params=feature_params,
            transform="zscore",
            rule="long_short_quantile",
        )
        self.name = f"feature_only_{feature}"

    def _apply(self, x, sigma):
        z = TRANSFORMS["zscore"](x)
        g = np.nansum(np.abs(z), axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.isfinite(z) & (g > 0), z / np.where(g > 0, g, 1.0), 0.0)
