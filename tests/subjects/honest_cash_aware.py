"""An honest test strategy that uses adjusted closes (split- and total-return), liquidity, its
own random stream, and its position, with a state carried between decisions."""

import numpy as np

from quant_research_engine.context import Strategy, TargetWeights


class HonestCashAware(Strategy):
    name = "honest_cash_aware"
    history_bars = 30

    def __init__(self, **params):
        super().__init__(**params)
        self._rng = None
        self._n = 0

    def decide(self, ctx):
        if self._rng is None:
            self._rng = ctx.rng("noise")
        self._n += 1
        ids = ctx.universe()
        a, _ = ctx.adjusted_close(21)
        tr, _ = ctx.adjusted_close(21, total_return=True)
        adv, sig = ctx.liquidity()
        if a.shape[0] < 21 or not ids:
            return TargetWeights({})
        with np.errstate(all="ignore"):
            mom = tr[-1] / tr[0] - 1 + 0.1 * (a[-1] / a[-5] - 1)
        z = (
            np.nan_to_num(mom)
            + 1e-3 * np.nan_to_num(sig)
            + 1e-4 * self._rng.standard_normal(len(ids))
        )
        held = np.array([ctx.portfolio().quantity(i) for i in ids])
        z = z - 1e-9 * np.sign(held) + 1e-6 * (self._n % 3)
        z = z - z.mean()
        g = np.abs(z).sum()
        return (
            TargetWeights({})
            if g == 0
            else TargetWeights({i: float(v / g) for i, v in zip(ids, z, strict=True)})
        )
