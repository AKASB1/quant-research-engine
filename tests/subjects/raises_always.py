"""A subject that fails in every world (reads something that does not exist)."""

from quant_research_engine.context import Strategy


class RaisesAlways(Strategy):
    name = "raises_always"
    history_bars = 1

    def decide(self, ctx):
        raise KeyError("QRE_STORE")
