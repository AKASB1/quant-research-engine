"""C8 overlapping_label_cv: chooses a momentum lookback by shuffled K-fold cross-validation on
labels that overlap (5-bar forward returns), so training labels share bars with test labels and
the cross-validation score is optimistic. Its decisions use only past data (G2 passes); the
validation guard G6 refuses the splitter unless allow_unpurged is given."""

import numpy as np

from quant_research_engine.context import Strategy, TargetWeights
from quant_research_engine.inference.splitters import make_splits

CANARY = {
    "id": "C8",
    "name": "overlapping_label_cv",
    "guard": "G6",
    "class": "OverlappingLabelCV",
    "kind": "cv",
    "validation": {"splitter": "shuffled_kfold", "k": 5, "horizon": 5},
}
LOOKBACKS = (5, 21, 63)
H = 5


class OverlappingLabelCV(Strategy):
    name = "overlapping_label_cv"
    history_bars = 260

    def __init__(self, **params):
        super().__init__(**params)
        self.lookback = None

    def _select(self, ctx, r):
        n = r.shape[0]
        lr = np.log1p(np.nan_to_num(r))
        c = np.cumsum(lr, axis=0)
        best, best_score = LOOKBACKS[0], -np.inf
        obs = np.arange(max(LOOKBACKS), n - H)
        if obs.size < 20:
            return best
        fwd = c[obs + H] - c[obs]
        splits = make_splits(
            "shuffled_kfold", obs.size, 5, H, rng=ctx.rng("cv"), allow_unpurged=True
        )
        for L in LOOKBACKS:
            sig = c[obs] - c[obs - L]
            score = 0.0
            for train, test in splits.folds:
                b = np.sign(np.nansum(sig[train] * fwd[train]))
                score += np.nansum(b * sig[test] * fwd[test])
            if score > best_score:
                best, best_score = L, score
        return best

    def decide(self, ctx):
        r, _ = ctx.returns(260)
        ids = ctx.universe()
        if r.shape[0] < 100 or not ids:
            return TargetWeights({})
        if self.lookback is None:
            self.lookback = self._select(ctx, r)
        m = np.nansum(r[-self.lookback :], axis=0)
        m = m - m.mean()
        g = np.abs(m).sum()
        return (
            TargetWeights({})
            if g == 0
            else TargetWeights({i: float(v / g) for i, v in zip(ids, m, strict=True)})
        )
