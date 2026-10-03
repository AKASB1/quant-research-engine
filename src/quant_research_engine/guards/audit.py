"""The audit: every guard on every canary and on every built-in strategy and feature.

Rows are subjects, columns the guards G1 to G7; a cell says ``caught`` or ``passed``:

- G1 (context truncation) holds by construction for every subject (its property test is check 4);
  canaries leak around the context, so the cell reads ``passed``.
- G2 replay audit: ``caught`` if any sampled instant shows a decision that differs between the
  real and a poisoned world (an exception at different bars counts), or if the subject changed
  the state of this package's modules or classes (undone after each world); ``error`` if the
  subject fails on the real store (it was not audited).
- G3 lint of the subject's module file (canary modules are exempt from enforcement; the cell
  still reports what the lint finds).
- G4 availability audit of the subject's panel feature (subjects without one: ``passed``).
- G5 fill timing: ``caught`` if the subject's run enables same-close fills (the run is then
  labelled UNSAFE and refused by the report generator).
- G6 validation splitters: ``caught`` if the subject's declared splitter violates the purge
  property for its label horizon, or if the subject created such splits while it was replayed.
- G7 universe: ``caught`` if a decision names an instrument outside the universe.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from quant_research_engine.backtest import RunConfig
from quant_research_engine.engine import EngineConfig
from quant_research_engine.guards.availability import availability_audit
from quant_research_engine.guards.canaries import load_subjects
from quant_research_engine.guards.lint import lint_file
from quant_research_engine.guards.poison import poison_store
from quant_research_engine.guards.replay import Subject, invariance_audit
from quant_research_engine.inference.splitters import UnpurgedSplitError, make_splits
from quant_research_engine.rng import stream
from quant_research_engine.strategies import BUILTIN_FEATURES, BUILTIN_STRATEGIES, FEATURE_ONLY

GUARDS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
HEADER = ["subject_id", "name", "kind", "named_guard", *GUARDS, "g2_instants", "g2_caught"]


def builtin_subjects() -> list[Subject]:
    out = []
    for sid, mod, cls, params in BUILTIN_STRATEGIES:
        out.append(Subject(sid, mod.rsplit(".", 1)[1], (mod, cls, params), kind="strategy"))
    for sid, fname, fparams in BUILTIN_FEATURES:
        out.append(
            Subject(
                sid,
                f"feature_{fname}",
                (FEATURE_ONLY[0], FEATURE_ONLY[1], {"feature": fname, "feature_params": fparams}),
                kind="feature",
            )
        )
    return out


def all_subjects(canary_dir: str | None) -> list[Subject]:
    subs = load_subjects(canary_dir) if canary_dir else []
    return subs + builtin_subjects()


def sample_instants(store, seed: int, n: int, warmup: int) -> list[int]:
    T = store.panel().shape[0]
    rng = stream(seed, "audit.instants")
    return sorted(rng.choice(np.arange(warmup, T), size=n, replace=False).tolist())


def poisoned_worlds(store, seed: int, instants: list[int]) -> dict:
    p = store.panel()
    return {k: poison_store(store, int(p.ts_event[k]), seed * 1_000_003 + k) for k in instants}


@dataclass
class SubjectResult:
    subject: Subject
    cells: dict = field(default_factory=dict)
    g2_flags: list = field(default_factory=list)  # (store seed, instant, caught)
    notes: list = field(default_factory=list)

    def row(self) -> list[str]:
        n = len(self.g2_flags)
        c = sum(1 for f in self.g2_flags if f[2])
        return [
            self.subject.id,
            self.subject.name,
            self.subject.kind,
            self.subject.named_guard or "-",
            *[self.cells[g] for g in GUARDS],
            str(n),
            str(c),
        ]


def _module_file(mod: str) -> str | None:
    spec = importlib.util.find_spec(mod)
    return spec.origin if spec else None


def _static_cells(s: Subject, store) -> dict:
    cells = {"G1": "passed"}
    f = _module_file(s.spec[0])
    cells["G3"] = "caught" if (f and lint_file(f)) else "passed"
    if s.feature is not None:
        res = availability_audit(s.feature(), store)
    elif s.kind == "feature":
        from quant_research_engine.features import make_feature

        p = s.spec[2]
        res = availability_audit(make_feature(p["feature"], **p.get("feature_params", {})), store)
    else:
        res = None
    cells["G4"] = "caught" if (res is not None and res.caught) else "passed"
    try:
        EngineConfig(**s.run_overrides).check()
        unsafe = EngineConfig(**s.run_overrides).unsafe
    except ValueError:
        unsafe = True
    cells["G5"] = "caught" if unsafe else "passed"
    if s.validation:
        v = s.validation
        try:
            make_splits(v["splitter"], 500, v.get("k", 5), v["horizon"], rng=stream(0, "cv"))
            cells["G6"] = "passed"
        except UnpurgedSplitError:
            cells["G6"] = "caught"
    else:
        cells["G6"] = "passed"
    return cells


def run_audit(
    subjects: list[Subject],
    stores: list[tuple[int, object]],
    n_instants: int,
    warmup: int = 20,
    rc: RunConfig | None = None,
) -> list[SubjectResult]:
    """``stores`` = [(seed, store)]; ``n_instants`` per store and subject."""
    rc = rc or RunConfig(
        engine=EngineConfig(initial_cash=1e6), rebalance_every=1, warmup_bars=warmup
    )
    results = {s.id: SubjectResult(s) for s in subjects}
    for s in subjects:
        results[s.id].cells.update(_static_cells(s, stores[0][1]))
        results[s.id].cells["G2"] = "passed"
        results[s.id].cells["G7"] = "passed"
    for seed, store in stores:
        inst = sample_instants(store, seed, n_instants, warmup)
        worlds = poisoned_worlds(store, seed, inst)
        for s in subjects:
            run_rc = rc
            if s.run_overrides:
                eng = rc.engine.model_copy(update={k: v for k, v in s.run_overrides.items()})
                run_rc = rc.model_copy(update={"engine": eng})
            for root in s.roots:
                if root not in sys.path:
                    sys.path.insert(0, root)
            r = invariance_audit(s, store, inst, run_rc, worlds)
            res = results[s.id]
            res.g2_flags.extend((seed, k, c) for k, c in zip(r.instants, r.caught, strict=True))
            if r.any_caught:
                res.cells["G2"] = "caught"
            if r.real_error is not None:
                res.cells["G2"] = "error"  # the subject fails on the real store: not audited
            if r.g7_violations:
                res.cells["G7"] = "caught"
            if r.unpurged_splits:
                res.cells["G6"] = "caught"  # observed, whether declared or not
            res.notes.extend(r.errors[:2])
            res.notes.extend(f"package state changed: {c}" for c in r.state_changes[:3])
    return [results[s.id] for s in subjects]


def table_rows(results: list[SubjectResult]) -> list[list[str]]:
    return [r.row() for r in results]


def write_table(path: str, results: list[SubjectResult]) -> bytes:
    lines = [",".join(HEADER)] + [",".join(r) for r in table_rows(results)]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def detection_curve(
    flags: list[bool], ms=(1, 2, 5, 10, 20, 50), draws: int = 100, seed: int = 0
) -> dict:
    """P(at least one caught instant among m drawn without replacement), from ``draws`` draws."""
    f = np.asarray(flags, dtype=bool)
    rng = stream(seed, "audit.instants.curve")
    out = {}
    for m in ms:
        if m > len(f):
            continue
        hits = 0
        for _ in range(draws):
            idx = rng.choice(len(f), size=m, replace=False)
            hits += bool(f[idx].any())
        out[m] = hits / draws
    return out
