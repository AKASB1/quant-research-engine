"""An honest test strategy: past-only reversal on the context, long-short demeaned weights.
It reads the portfolio and the open orders too, so the audit replays them as well."""

import numpy as np

from quant_research_engine.context import Strategy, TargetWeights


class HonestReversal(Strategy):
    name = "honest_reversal"
    history_bars = 10
    series_suffixes = (".signal_x",)

    def decide(self, ctx):
        r, _ = ctx.returns(5)
        ids = ctx.universe()
        if r.shape[0] < 5 or not ids:
            return TargetWeights({})
        s = -np.nansum(r, axis=0)
        x = np.array([ctx.series(f"{i}.signal_x")[1][-1:].sum() for i in ids])
        s = s + 1e-3 * x + 1e-12 * ctx.portfolio().cash + 1e-15 * len(ctx.open_orders())
        s = s - s.mean()
        g = np.abs(s).sum()
        if g == 0:
            return TargetWeights({})
        return TargetWeights({i: float(v / g) for i, v in zip(ids, s, strict=True)})
