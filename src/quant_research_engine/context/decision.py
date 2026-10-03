"""Decisions a strategy returns, and the strategy base class."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetWeights:
    """Fractions of the sizing base per instrument; instruments not named have target 0."""

    weights: dict[str, float]


@dataclass(frozen=True)
class TargetShares:
    """Target quantities per instrument (in the shares of the decision instant)."""

    shares: dict[str, float]


@dataclass(frozen=True)
class OrderSpec:
    instrument_id: str
    quantity: float
    tif: str = "day"


@dataclass(frozen=True)
class Orders:
    orders: tuple[OrderSpec, ...] = field(default_factory=tuple)


Decision = TargetWeights | TargetShares | Orders


def decision_key(d: Decision | None) -> tuple:
    """Canonical, bit-exact form of a decision (floats as their hexadecimal representation)."""
    if d is None:
        return ("none",)
    if isinstance(d, TargetWeights):
        return ("w",) + tuple((k, float(v).hex()) for k, v in sorted(d.weights.items()))
    if isinstance(d, TargetShares):
        return ("s",) + tuple((k, float(v).hex()) for k, v in sorted(d.shares.items()))
    if isinstance(d, Orders):
        return ("o",) + tuple((o.instrument_id, float(o.quantity).hex(), o.tif) for o in d.orders)
    raise TypeError(f"not a decision: {type(d).__name__}")


def decision_instruments(d: Decision | None) -> list[str]:
    if d is None:
        return []
    if isinstance(d, TargetWeights):
        return [k for k, v in d.weights.items() if v != 0]
    if isinstance(d, TargetShares):
        return [k for k, v in d.shares.items() if v != 0]
    return [o.instrument_id for o in d.orders]


def weights_from_array(ids, values) -> TargetWeights:
    """Weights from aligned ids and values; NaN and zero entries are left out."""
    out = {}
    for k, v in zip(ids, values, strict=True):
        v = float(v)
        if v != 0.0 and not math.isnan(v):
            out[str(k)] = v
    return TargetWeights(out)


class Strategy:
    """Base class. A strategy has a ``name``, JSON ``params``, and ``decide(ctx) -> Decision``.

    It is deterministic given its parameters and its random streams (``ctx.rng``), reads no
    clock, and keeps only private state. ``history_bars`` bounds the history the context copies
    (None: every known bar); ``series_suffixes`` names the per-instrument series it reads
    (for example ``(".signal_x",)``).
    """

    name: str = "strategy"
    history_bars: int | None = None
    series_suffixes: tuple[str, ...] = ()

    def __init__(self, **params: Any):
        self.params = dict(params)

    def decide(self, ctx):  # pragma: no cover - interface
        raise NotImplementedError
