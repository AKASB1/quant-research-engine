"""signal_x_ewma(10), z-scored, vol_target(scale 0.01) with 63-bar volatility."""

from quant_research_engine.strategies.base import FeatureRuleStrategy


class SignalXVolTarget(FeatureRuleStrategy):
    """signal_x_ewma(10), z-scored, vol_target(scale 0.01) with 63-bar volatility."""

    def __init__(self, halflife: float = 10, scale: float = 0.01, vol_lookback: int = 63):
        super().__init__(
            feature="signal_x_ewma",
            feature_params={"halflife": halflife},
            rule="vol_target",
            rule_params={"scale": scale},
            vol_lookback=vol_lookback,
        )
        self.params = {"halflife": halflife, "scale": scale, "vol_lookback": vol_lookback}
        self.name = "signal_x_voltarget"
