"""C4 survivors_only: trades only the instruments that never delist in the whole store (a
survivorship-biased universe). Guard: G2 (the poison flips which instruments delist)."""

from _peek_helper import handle

from quant_research_engine.context import Strategy, TargetWeights

CANARY = {"id": "C4", "name": "survivors_only", "guard": "G2", "class": "SurvivorsOnly"}


class SurvivorsOnly(Strategy):
    name = "survivors_only"
    history_bars = 1

    def decide(self, ctx):
        ins = handle().tables["instruments"]
        survivors = {
            str(i) for i, d in zip(ins["instrument_id"], ins["ts_delist"], strict=True) if d == -1
        }
        uni = [i for i in ctx.universe() if i in survivors]
        if not uni:
            return TargetWeights({})
        return TargetWeights({i: 1.0 / len(uni) for i in uni})
