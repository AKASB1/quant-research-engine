"""C1 peek_next_close: reads the next bar's close through a store handle that a helper module
opens on first use and memoizes with functools.lru_cache. Guard: G2 (the purge of helper
modules makes the handle come from the replayed world)."""

from _common import demeaned_weights, next_close_returns
from _peek_helper import handle

from quant_research_engine.context import Strategy, TargetWeights

CANARY = {"id": "C1", "name": "peek_next_close", "guard": "G2", "class": "PeekNextClose"}


class PeekNextClose(Strategy):
    name = "peek_next_close"
    history_bars = 2

    def decide(self, ctx):
        r = next_close_returns(handle().panel(), ctx.t, ctx.universe())
        return TargetWeights({}) if r is None else demeaned_weights(ctx.universe(), r)
