"""Volume trend, long only: volume_trend(21), z-scored, long_only_topk(5)."""

from quant_research_engine.strategies.base import FeatureRuleStrategy


class VolumeTrendTopK(FeatureRuleStrategy):
    """Volume trend, long only: volume_trend(21), z-scored, long_only_topk(5)."""

    def __init__(self, lookback: int = 21, k: int = 5):
        super().__init__(
            feature="volume_trend",
            feature_params={"lookback": lookback},
            rule="long_only_topk",
            rule_params={"k": k},
        )
        self.params = {"lookback": lookback, "k": k}
        self.name = "volume_trend_topk"
