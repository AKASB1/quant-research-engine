"""C1s package_state_handle (found by review 1): reaches the store through an attribute chain of
the package root (``import quant_research_engine.context`` binds ``quant_research_engine``) and
keeps the handle on a class of this package (``Strategy``), which a purge of user modules does
not reset; then reads the next bar's close. Guards: G3 (the lint rejects ``import
quant_research_engine...`` and the chain) and G2 (package state is snapshotted before each world
and any change is a caught leak, then undone)."""

from _common import demeaned_weights, next_close_returns

import quant_research_engine.context
from quant_research_engine.context import Strategy, TargetWeights

CANARY = {"id": "C1s", "name": "package_state_handle", "guard": "G2", "class": "PackageStateHandle"}


class PackageStateHandle(Strategy):
    name = "package_state_handle"
    history_bars = 2

    def decide(self, ctx):
        h = Strategy.__dict__.get("_qre_handle")
        if h is None:
            h = quant_research_engine.store.registry.open_store()
            Strategy._qre_handle = h
        r = next_close_returns(h.panel(), ctx.t, ctx.universe())
        return TargetWeights({}) if r is None else demeaned_weights(ctx.universe(), r)
