"""C6 early_release_feature: a panel feature that places the revised series fund_x at its
period end (ts_event) instead of at its release, and declares availability at the period end.
Its context implementation reads the as-of value, so the replay audit sees nothing; the
availability audit G4 compares the declared availability with the inputs' ts_avail."""

import numpy as np

from quant_research_engine.features import Feature
from quant_research_engine.strategies.base import FeatureOnly

CANARY = {
    "id": "C6",
    "name": "early_release_feature",
    "guard": "G4",
    "class": "EarlyReleaseStrategy",
    "kind": "feature",
    "feature": "EarlyReleaseFeature",
}


class EarlyReleaseFeature(Feature):
    name = "early_release_fund_x"
    inputs = ("series:.fund_x",)
    series_suffixes = (".fund_x",)

    @property
    def lookback(self):
        return 0

    def compute(self, ctx):
        out = []
        for iid in ctx.universe():
            try:
                _, val = ctx.series(f"{iid}.fund_x")
            except KeyError:
                val = np.zeros(0)
            out.append(val[-1] if len(val) else np.nan)
        return np.asarray(out, dtype=np.float64)

    def panel(self, src):
        return src.series_by_event(".fund_x")  # the mistake: value at the period end

    def declared_availability(self, src):
        return src.declared_bar_avail()  # declares the period end


class EarlyReleaseStrategy(FeatureOnly):
    def __init__(self):
        super().__init__(feature="momentum")
        self.feature = EarlyReleaseFeature()
        self.series_suffixes = (".fund_x",)
        self.history_bars = 2
        self.name = "early_release_feature"
