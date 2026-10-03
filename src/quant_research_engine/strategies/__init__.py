"""Built-in strategies (one feature composed with one portfolio rule) and the audit registry.

``BUILTIN_STRATEGIES`` and ``BUILTIN_FEATURES`` are the subjects the ``audit`` command checks:
(id, module, class, params). The 108 configurations of the calibration experiment are parameter
choices of these classes, not separate audit subjects.
"""

BUILTIN_STRATEGIES = [
    ("S_momentum_ls", "quant_research_engine.strategies.momentum_ls", "MomentumLS", {}),
    ("S_reversal_ls", "quant_research_engine.strategies.reversal_ls", "ReversalLS", {}),
    ("S_low_vol_ls", "quant_research_engine.strategies.low_vol_ls", "LowVolLS", {}),
    ("S_signal_x_ls", "quant_research_engine.strategies.signal_x_ls", "SignalXLS", {}),
    (
        "S_volume_trend_topk",
        "quant_research_engine.strategies.volume_trend_topk",
        "VolumeTrendTopK",
        {},
    ),
    (
        "S_signal_x_voltarget",
        "quant_research_engine.strategies.signal_x_voltarget",
        "SignalXVolTarget",
        {},
    ),
]

BUILTIN_FEATURES = [
    ("F_momentum", "momentum", {"lookback": 21, "skip": 5}),
    ("F_reversal", "reversal", {"lookback": 5}),
    ("F_low_vol", "low_vol", {"lookback": 63}),
    ("F_volume_trend", "volume_trend", {"lookback": 21}),
    ("F_signal_x_ewma", "signal_x_ewma", {"halflife": 3}),
]

FEATURE_ONLY = ("quant_research_engine.strategies.base", "FeatureOnly")
