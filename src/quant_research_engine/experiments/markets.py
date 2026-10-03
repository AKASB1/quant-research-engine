"""Synthetic markets of the experiments: a seed of experiment ``<id>`` fixes the whole market
through the derived seed ``stream(seed, "experiment.<id>")``, so every strategy sees the same
data and quick runs of different experiments never share a market."""

from __future__ import annotations

import os

from quant_research_engine.rng import derived_seed
from quant_research_engine.synth import generate, load_synth_config


def synth_config(name: str, **overrides):
    from quant_research_engine.experiments.runner import repo_root

    cfg = load_synth_config(os.path.join(repo_root(), "configs", "synth", f"{name}.json"))
    return cfg.with_overrides(**overrides) if overrides else cfg


def market_seed(exp_id: str, seed: int) -> int:
    return derived_seed(int(seed), f"experiment.{exp_id}")


def market(exp_id: str, name: str, seed: int, **overrides):
    return generate(synth_config(name, **overrides), market_seed(exp_id, seed))
