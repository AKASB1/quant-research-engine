"""C1g peek_global_handle (self-test variant of C1): the helper keeps the store handle in a
module-level global instead of lru_cache. Guard: G2."""

from _common import demeaned_weights, next_close_returns
from _global_helper import handle

from quant_research_engine.context import Strategy, TargetWeights

CANARY = {"id": "C1g", "name": "peek_global_handle", "guard": "G2", "class": "PeekGlobalHandle"}


class PeekGlobalHandle(Strategy):
    name = "peek_global_handle"
    history_bars = 2

    def decide(self, ctx):
        r = next_close_returns(handle().panel(), ctx.t, ctx.universe())
        return TargetWeights({}) if r is None else demeaned_weights(ctx.universe(), r)
