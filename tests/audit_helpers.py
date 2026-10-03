"""Shared setup for audit tests: one `small` store, its poisoned copies, and the run config."""

from __future__ import annotations

import os

import numpy as np

from quant_research_engine.backtest import RunConfig
from quant_research_engine.engine import EngineConfig
from quant_research_engine.guards.poison import poison_store
from quant_research_engine.rng import stream
from quant_research_engine.synth import generate, load_synth_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANARIES = os.path.join(ROOT, "tests", "canaries")
SUBJECTS = os.path.join(ROOT, "tests", "subjects")
_CACHE: dict = {}


def audit_world(seed: int = 1, n_instants: int = 50, warmup: int = 20):
    key = (seed, n_instants, warmup)
    if key not in _CACHE:
        st, _ = generate(
            load_synth_config(os.path.join(ROOT, "configs", "synth", "small.json")), seed
        )
        p = st.panel()
        T = p.shape[0]
        rng = stream(seed, "audit.instants")
        inst = sorted(rng.choice(np.arange(warmup, T), size=n_instants, replace=False).tolist())
        pois = {k: poison_store(st, int(p.ts_event[k]), seed * 1_000_003 + k) for k in inst}
        rc = RunConfig(engine=EngineConfig(initial_cash=1e6), rebalance_every=1, warmup_bars=warmup)
        _CACHE[key] = (st, inst, pois, rc)
    return _CACHE[key]
