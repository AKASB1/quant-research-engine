"""Selection-aware inference: PSR, DSR, PBO, bootstrap, splitters, trial registry."""

from quant_research_engine.inference.bootstrap import (
    bootstrap_stat,
    percentile_interval,
    stationary_indices,
)
from quant_research_engine.inference.pbo import pbo_cscv
from quant_research_engine.inference.registry import TrialRegistry
from quant_research_engine.inference.sharpe import dsr, expected_max_sharpe, psr
from quant_research_engine.inference.splitters import (
    UnpurgedSplitError,
    cpcv,
    make_splits,
    purged_kfold,
    shuffled_kfold,
    walk_forward,
)

__all__ = [
    "TrialRegistry",
    "UnpurgedSplitError",
    "bootstrap_stat",
    "cpcv",
    "dsr",
    "expected_max_sharpe",
    "make_splits",
    "pbo_cscv",
    "percentile_interval",
    "psr",
    "purged_kfold",
    "shuffled_kfold",
    "stationary_indices",
    "walk_forward",
]
