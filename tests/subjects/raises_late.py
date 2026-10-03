"""An honest subject that raises only after 200 known sessions (context data only)."""

import numpy as np

from quant_research_engine.context import Strategy, TargetWeights


class RaisesLate(Strategy):
    name = "raises_late"

    def decide(self, ctx):
        _, te = ctx.bars("close", 10**6)
        if len(te) >= 200:
            raise RuntimeError("late failure")
        r, _ = ctx.returns(3)
        ids = ctx.universe()
        if r.shape[0] < 3 or not ids:
            return TargetWeights({})
        s = -np.nansum(r, axis=0)
        s = s - s.mean()
        g = np.abs(s).sum()
        return (
            TargetWeights({})
            if g == 0
            else TargetWeights({i: float(v / g) for i, v in zip(ids, s, strict=True)})
        )
