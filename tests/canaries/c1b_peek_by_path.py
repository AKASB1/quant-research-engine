"""C1b peek_by_path: opens the store by path inside decide (the real store's path in every
world, like a hard-coded path) and reads the next bar's close. Guard: G2 (``open_store`` with
any path returns the replayed world's store)."""

import os

from _common import demeaned_weights, next_close_returns

from quant_research_engine.context import Strategy, TargetWeights
from quant_research_engine.store.registry import open_store

CANARY = {"id": "C1b", "name": "peek_by_path", "guard": "G2", "class": "PeekByPath"}


class PeekByPath(Strategy):
    name = "peek_by_path"
    history_bars = 2

    def decide(self, ctx):
        panel = open_store(os.environ.get("QRE_STORE")).panel()
        r = next_close_returns(panel, ctx.t, ctx.universe())
        return TargetWeights({}) if r is None else demeaned_weights(ctx.universe(), r)
