"""C2 full_sample_mean: computes in its constructor, from the whole store, each instrument's
mean return over the entire sample and goes long the highest (a leak cached at construction).
Guard: G2 (the strategy is instantiated again in every world)."""

import os
import warnings

import numpy as np

from quant_research_engine.context import Strategy, TargetWeights
from quant_research_engine.store.registry import open_store

CANARY = {"id": "C2", "name": "full_sample_mean", "guard": "G2", "class": "FullSampleMean"}


class FullSampleMean(Strategy):
    name = "full_sample_mean"
    history_bars = 1

    def __init__(self, **params):
        super().__init__(**params)
        p = open_store(os.environ.get("QRE_STORE")).panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # instruments without returns
            means = np.nanmean(p.ret, axis=0)
        order = np.argsort(-np.nan_to_num(means, nan=-np.inf), kind="stable")
        self.ranking = [p.ids[j] for j in order.tolist()]

    def decide(self, ctx):
        uni = set(ctx.universe())
        for iid in self.ranking:
            if iid in uni:
                return TargetWeights({iid: 1.0})
        return TargetWeights({})
