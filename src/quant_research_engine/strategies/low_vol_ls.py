"""Low-volatility long-short: low_vol(63), z-scored, long_short_quantile(0.3, signal)."""

from quant_research_engine.strategies.base import FeatureRuleStrategy


class LowVolLS(FeatureRuleStrategy):
    """Low-volatility long-short: low_vol(63), z-scored, long_short_quantile(0.3, signal)."""

    def __init__(
        self, lookback: int = 63, q: float = 0.3, weighting: str = "signal", gross: float = 1.0
    ):
        super().__init__(
            feature="low_vol",
            feature_params={"lookback": lookback},
            rule="long_short_quantile",
            rule_params={"q": q, "weighting": weighting, "gross": gross},
        )
        self.params = {"lookback": lookback, "q": q, "weighting": weighting, "gross": gross}
        self.name = "low_vol_ls"
