"""Backtest runner: decision schedule, sizing, and the glue between a strategy and an engine.

``run_strategy`` walks the store's calendar with the event engine; at each scheduled decision
bar it builds a :class:`DecisionContext` (copies of rows known at the bar's close), calls
``strategy.decide``, and submits the decision. The run happens inside ``registry.world(store)``,
so a strategy that opens a store by path gets the store under test.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from quant_research_engine.context.context import ContextBuilder
from quant_research_engine.engine.event import EngineConfig, EventEngine, EventResult
from quant_research_engine.store import registry


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine: EngineConfig = Field(default_factory=EngineConfig)
    rebalance_every: int = Field(default=1, ge=1)
    warmup_bars: int = Field(default=0, ge=0)
    decision_bars: list[int] | None = None
    strategy_seed: int = 0


def schedule(n_bars: int, rc: RunConfig) -> list[int]:
    """Calendar indices of the decisions: every ``rebalance_every`` bars from ``warmup_bars``,
    or the bars named in ``decision_bars``."""
    if rc.decision_bars is not None:
        return sorted({int(k) for k in rc.decision_bars if 0 <= int(k) < n_bars})
    return list(range(rc.warmup_bars, n_bars, rc.rebalance_every))


@dataclass
class RunOutput:
    result: EventResult
    decisions: list = field(default_factory=list)  # (k, decision, universe)


def instantiate(spec):
    """Import ``spec[0]``, take class ``spec[1]``, and build it with the params ``spec[2]``."""
    module, cls, params = spec
    mod = importlib.import_module(module)
    return getattr(mod, cls)(**dict(params or {}))


def run_strategy(
    store,
    strategy,
    rc: RunConfig,
    *,
    stop_after: int | None = None,
    record: list | None = None,
    progress: dict | None = None,
) -> RunOutput:
    """``record`` receives (bar, decision, universe) as decisions are made and ``progress["k"]``
    the bar being processed, so a caller keeps both when the strategy raises."""
    p = store.panel()
    T = p.shape[0]
    eng = EventEngine(
        p, rc.engine, getattr(strategy, "name", "strategy"), calendar=store.calendar_name
    )
    builder = ContextBuilder(
        store,
        getattr(strategy, "name", "strategy"),
        rc.strategy_seed,
        getattr(strategy, "history_bars", None),
        tuple(getattr(strategy, "series_suffixes", ())),
    )
    sched = set(schedule(T, rc))
    decisions = record if record is not None else []
    progress = progress if progress is not None else {}
    with registry.world(store):
        for k in range(T):
            progress["k"] = k
            eng.process_bar(k)
            if k in sched:
                ctx = builder.build(
                    int(p.ts_event[k]), eng.portfolio_view(), eng.open_orders_view()
                )
                d = strategy.decide(ctx)
                uni = list(ctx.universe())
                decisions.append((k, d, uni))
                eng.submit(d, uni)
                if eng.cfg.unsafe:
                    eng.fill_same_close()
            if stop_after is not None and k >= stop_after:
                break
    return RunOutput(eng.result(), decisions)
