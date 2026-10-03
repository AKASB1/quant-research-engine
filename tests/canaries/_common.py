"""Shared weighting for canaries (they leak on purpose; the rule itself is ordinary)."""

import numpy as np

from quant_research_engine.context import TargetWeights


def demeaned_weights(ids, values) -> TargetWeights:
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return TargetWeights({})
    x = np.where(ok, v - v[ok].mean(), 0.0)
    g = np.abs(x).sum()
    if g == 0:
        return TargetWeights({})
    return TargetWeights({i: float(w) for i, w in zip(ids, x / g, strict=True) if w != 0.0})


def bar_index(panel, t) -> int:
    return int(np.searchsorted(panel.ts_event, t, side="right")) - 1


def next_close_returns(panel, t, ids):
    k = bar_index(panel, t)
    if k + 1 >= len(panel.ts_event) or not ids:
        return None
    pos = {iid: j for j, iid in enumerate(panel.ids)}
    cols = [pos[i] for i in ids]
    with np.errstate(all="ignore"):
        r = panel.close[k + 1, cols] / panel.close[k, cols] - 1.0
    return np.where(np.isfinite(r), r, np.nan)
