"""C8 without its declaration: selects a lookback by shuffled K-fold on overlapping labels but
declares no validation set-up; the audit must observe the unpurged splits (G6)."""

import numpy as np

from quant_research_engine.context import Strategy, TargetWeights
from quant_research_engine.inference.splitters import make_splits


class UndeclaredCV(Strategy):
    name = "undeclared_cv"
    history_bars = 120

    def decide(self, ctx):
        r, _ = ctx.returns(120)
        ids = ctx.universe()
        if r.shape[0] < 60 or not ids:
            return TargetWeights({})
        make_splits("shuffled_kfold", r.shape[0] - 5, 5, 5, rng=ctx.rng("cv"), allow_unpurged=True)
        m = np.nansum(r[-21:], axis=0)
        m = m - m.mean()
        g = np.abs(m).sum()
        return (
            TargetWeights({})
            if g == 0
            else TargetWeights({i: float(v / g) for i, v in zip(ids, m, strict=True)})
        )
