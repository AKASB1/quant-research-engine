"""Momentum long-short: momentum(21, 5), z-scored, long_short_quantile(0.2, equal)."""

from quant_research_engine.strategies.base import FeatureRuleStrategy


class MomentumLS(FeatureRuleStrategy):
    """Momentum long-short: momentum(21, 5), z-scored, long_short_quantile(0.2, equal)."""

    def __init__(
        self,
        lookback: int = 21,
        skip: int = 5,
        q: float = 0.2,
        weighting: str = "equal",
        gross: float = 1.0,
    ):
        super().__init__(
            feature="momentum",
            feature_params={"lookback": lookback, "skip": skip},
            rule="long_short_quantile",
            rule_params={"q": q, "weighting": weighting, "gross": gross},
        )
        self.params = {
            "lookback": lookback,
            "skip": skip,
            "q": q,
            "weighting": weighting,
            "gross": gross,
        }
        self.name = "momentum_ls"
