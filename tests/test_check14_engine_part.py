"""Check 14 (engine part): the fundamental-law weights, scaled to gross exposure 1, through the
event engine with next_open fills, equity sizing, a decision on every bar, zero costs, and
no participation cap earn a pooled Sharpe ratio within 15 % of the bound times
1 - gap_share * (1 - phi). The weights use the latent signal: an oracle, used only to check
the engine against the generator."""

import math
import os

import numpy as np
import pytest

from quant_research_engine.backtest import RunConfig, run_strategy
from quant_research_engine.context import Strategy, TargetWeights
from quant_research_engine.costs import zero_costs
from quant_research_engine.engine import EngineConfig
from quant_research_engine.synth import generate, load_synth_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class OracleWeights(Strategy):
    """Target weights s_t / sum|s_t| from the generator's latent signal (an oracle)."""

    name = "oracle_fl"
    history_bars = 1

    def __init__(self, s, ts_event, ids):
        super().__init__()
        self._w = s / np.sum(np.abs(s), axis=1, keepdims=True)
        self._k = {int(t): k for k, t in enumerate(ts_event.tolist())}
        self._ids = ids

    def decide(self, ctx):
        k = self._k[ctx.t]
        return TargetWeights({i: float(self._w[k, j]) for j, i in enumerate(self._ids)})


@pytest.mark.slow
def test_engine_earns_the_bound_times_the_overnight_factor():
    cfg = load_synth_config(os.path.join(ROOT, "configs", "synth", "planted_clean.json"))
    rets = []
    for seed in range(1, 21):
        st, tr = generate(cfg, seed)
        p = st.panel()
        strat = OracleWeights(tr.s, p.ts_event, p.ids)
        # the participation cap is a physical limit, not a cost: off here, because equity compounds
        costs = zero_costs().model_copy(update={"participation_cap": 1e9})
        rc = RunConfig(
            engine=EngineConfig(initial_cash=1e6, costs=costs, sizing="equity"), warmup_bars=0
        )
        out = run_strategy(st, strat, rc)
        assert out.result.cap_binds == 0
        eq = out.result.equity
        rets.append(eq[2:] / eq[1:-1] - 1.0)  # from the first bar that holds a position
    pooled = np.concatenate(rets)
    sr = pooled.mean() / pooled.std(ddof=1)
    bound = cfg.ic * math.sqrt(cfg.n_instruments)
    target = bound * (1 - cfg.gap_share * (1 - cfg.persistence))
    assert abs(sr - target) <= 0.15 * target, (sr, target)
