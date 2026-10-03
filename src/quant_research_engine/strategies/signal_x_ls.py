"""Alternative-data long-short: signal_x_ewma(3), z-scored, long_short_quantile(0.2, equal)."""

from quant_research_engine.strategies.base import FeatureRuleStrategy


class SignalXLS(FeatureRuleStrategy):
    """Alternative-data long-short: signal_x_ewma(3), z-scored, long_short_quantile(0.2, equal)."""

    def __init__(
        self, halflife: float = 3, q: float = 0.2, weighting: str = "equal", gross: float = 1.0
    ):
        super().__init__(
            feature="signal_x_ewma",
            feature_params={"halflife": halflife},
            rule="long_short_quantile",
            rule_params={"q": q, "weighting": weighting, "gross": gross},
        )
        self.params = {"halflife": halflife, "q": q, "weighting": weighting, "gross": gross}
        self.name = "signal_x_ls"
