"""Check 5 (strategy side): honest strategies are invariant under future perturbation at 100
random decision instants on the `small` store (stream audit.instants). The built-in strategies
and features are audited in test_check06_audit_table.py and test_check05_features.py."""

import pytest
from audit_helpers import SUBJECTS, audit_world

from quant_research_engine.guards.replay import Subject, invariance_audit

HONEST = [
    Subject("H1", "honest_reversal", ("honest_reversal", "HonestReversal", {}), roots=[SUBJECTS]),
    Subject(
        "H2", "honest_cash_aware", ("honest_cash_aware", "HonestCashAware", {}), roots=[SUBJECTS]
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("subject", HONEST, ids=[s.id for s in HONEST])
def test_honest_strategies_pass_100_instants(subject):
    import sys

    if SUBJECTS not in sys.path:
        sys.path.insert(0, SUBJECTS)
    st, inst, pois, rc = audit_world(seed=1, n_instants=100)
    r = invariance_audit(subject, st, inst, rc, pois)
    assert len(r.instants) == 100
    assert not r.any_caught, (r.first_diff, r.errors)
    assert r.g7_violations == 0
    assert not r.errors
