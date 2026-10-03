"""Self-test of the replay audit (part of check 6): the leaks that the isolation mechanism exists
for are caught, and switching that mechanism off would let them pass."""

import sys

import pytest
from audit_helpers import CANARIES, audit_world

from quant_research_engine.guards import replay
from quant_research_engine.guards.canaries import load_subjects
from quant_research_engine.guards.replay import invariance_audit
from quant_research_engine.store import registry


def _subjects():
    return {s.id: s for s in load_subjects(CANARIES)}


@pytest.mark.slow
@pytest.mark.parametrize("cid", ["C1", "C1g", "C1b", "C2", "C3", "C4", "C5"])
def test_g2_catches_canary(cid):
    st, inst, pois, rc = audit_world(seed=1, n_instants=20)
    s = _subjects()[cid]
    assert s.named_guard == "G2"
    r = invariance_audit(s, st, inst, rc, pois)
    assert r.any_caught, cid


@pytest.mark.slow
def test_without_the_purge_memoized_helpers_would_pass(monkeypatch):
    """Re-importing only the canary module (no purge of helper modules) misses C1 and C1g."""
    st, inst, pois, rc = audit_world(seed=1, n_instants=20)
    subs = _subjects()

    def naive_purge(subject_module, roots, baseline):
        sys.modules.pop(subject_module, None)
        return [subject_module]

    for name in ("_peek_helper", "_global_helper"):
        sys.modules.pop(name, None)
    monkeypatch.setattr(replay, "purge_modules", naive_purge)
    for cid in ("C1", "C1g"):
        r = invariance_audit(subs[cid], st, inst, rc, pois)
        assert not r.any_caught, cid
    for name in ("_peek_helper", "_global_helper"):
        sys.modules.pop(name, None)


@pytest.mark.slow
def test_without_the_registry_override_a_read_by_path_would_pass(monkeypatch, tmp_path):
    """If open_store ignored the world and read the real store's path, C1b would pass."""
    from quant_research_engine.store import write_store

    st, inst, pois, rc = audit_world(seed=1, n_instants=10)
    path = str(tmp_path / "real")
    write_store(path, st)
    monkeypatch.setenv("QRE_STORE", path)
    real_open = registry.read_store

    def by_path_only(p=None):
        return real_open(p or path)

    s = _subjects()["C1b"]
    monkeypatch.setattr(registry, "open_store", by_path_only)
    sys.modules.pop("c1b_peek_by_path", None)
    monkeypatch.setattr(replay, "purge_modules", lambda m, r, b: [sys.modules.pop(m, None)])
    r = invariance_audit(s, st, inst[:5], rc, pois)
    assert not r.any_caught
    sys.modules.pop("c1b_peek_by_path", None)
