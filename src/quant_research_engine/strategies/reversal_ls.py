"""Short-term reversal long-short: reversal(5), z-scored, long_short_quantile(0.2, equal)."""

from quant_research_engine.strategies.base import FeatureRuleStrategy


class ReversalLS(FeatureRuleStrategy):
    """Short-term reversal long-short: reversal(5), z-scored, long_short_quantile(0.2, equal)."""

    def __init__(
        self, lookback: int = 5, q: float = 0.2, weighting: str = "equal", gross: float = 1.0
    ):
        super().__init__(
            feature="reversal",
            feature_params={"lookback": lookback},
            rule="long_short_quantile",
            rule_params={"q": q, "weighting": weighting, "gross": gross},
        )
        self.params = {"lookback": lookback, "q": q, "weighting": weighting, "gross": gross}
        self.name = "reversal_ls"
