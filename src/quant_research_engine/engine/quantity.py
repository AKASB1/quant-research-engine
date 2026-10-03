"""Quantity rounding shared by both engines and the sizing layer.

Quantities are rounded toward zero to a multiple of the lot size (when positive) and held on a
binary grid of 2^-20 units, so that sums, differences, and rescales by integer split ratios are
exact in float64 (a deviation of at most about 1e-6 units from the unrounded value).
"""

from __future__ import annotations

import math

import numpy as np

GRID = float(2**20)


def round_lot(x, lot):
    """Round toward zero to a multiple of ``lot`` (if > 0), then toward zero to the grid."""
    x = np.asarray(x, dtype=np.float64)
    lot = np.asarray(lot, dtype=np.float64)
    safe = np.where(lot > 0, lot, 1.0)
    y = np.where(lot > 0, np.trunc(x / safe) * safe, x)
    out = np.trunc(y * GRID) / GRID
    return out + 0.0  # no negative zero


def snap(x):
    """Round to the nearest grid point (after a split rescale)."""
    x = np.asarray(x, dtype=np.float64)
    return np.round(x * GRID) / GRID + 0.0


def round_lot_scalar(x: float, lot: float) -> float:
    """Scalar form of :func:`round_lot` (same arithmetic: exact truncations, power-of-two grid)."""
    y = math.trunc(x / lot) * lot if lot > 0 else x
    return math.trunc(y * GRID) / GRID + 0.0


def snap_scalar(x: float) -> float:
    return round(x * GRID) / GRID + 0.0
