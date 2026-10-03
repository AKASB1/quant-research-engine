"""Check 6: all canaries are caught by the guard named for them, no built-in strategy or feature
raises an alarm, and the audit table of the test run (one `small` store of seed 1, 50 sampled
instants per subject, stream audit.instants) equals the committed tests/data/audit_table.csv."""

import os

import pytest
from audit_helpers import CANARIES, ROOT

from quant_research_engine.guards.audit import GUARDS, all_subjects, run_audit, table_rows
from quant_research_engine.synth import generate, load_synth_config

EXPECTED = {
    "C1": "G2",
    "C1b": "G2",
    "C1g": "G2",
    "C1s": "G2",
    "C2": "G2",
    "C3": "G2",
    "C4": "G2",
    "C5": "G2",
    "C6": "G4",
    "C7": "G5",
    "C8": "G6",
}


@pytest.mark.slow
def test_audit_table_matches_committed_and_every_canary_is_caught():
    st, _ = generate(load_synth_config(os.path.join(ROOT, "configs", "synth", "small.json")), 1)
    res = run_audit(all_subjects(CANARIES), [(1, st)], 50, 20)
    rows = table_rows(res)
    named = {r.subject.id: r.subject.named_guard for r in res}
    for cid, guard in EXPECTED.items():
        assert named[cid] == guard
    for r in res:
        if r.subject.named_guard:
            assert r.cells[r.subject.named_guard] == "caught", r.subject.id
        else:
            assert all(r.cells[g] == "passed" for g in GUARDS), r.subject.id
        assert "error" not in r.cells.values(), r.subject.id
    lines = [",".join(x) for x in rows]
    with open(os.path.join(ROOT, "tests", "data", "audit_table.csv"), encoding="utf-8") as f:
        committed = f.read().split("\n")[1:-1]
    assert lines == committed
