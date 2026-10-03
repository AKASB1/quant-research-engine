"""Guard G4: availability audit of panel features.

A feature declares its inputs and lookback; the framework records the ``ts_avail`` of every
input row the panel implementation reads and derives a value's availability as the latest
``ts_avail`` of the inputs in its lookback window. A feature whose declared availability is
earlier than that (a release lag ignored) fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_research_engine.features import PanelSource, derived_availability


@dataclass
class AvailabilityResult:
    feature: str
    cells: int
    violations: int

    @property
    def caught(self) -> bool:
        return self.violations > 0


def availability_audit(feature, store) -> AvailabilityResult:
    src = PanelSource(store, track=True)
    vals = feature.panel(src)
    derived = derived_availability(src.used, feature.lookback)
    declared = feature.declared_availability(src)
    finite = np.isfinite(vals)
    bad = finite & (declared < derived)
    return AvailabilityResult(feature.label, int(finite.sum()), int(bad.sum()))
