"""Replay audit (guard G2, future-perturbation invariance) and the universe check (G7).

For each sampled decision instant ``t`` the audit replays the strategy from its first decision
up to and including ``t`` in two worlds, the real store and a copy poisoned after ``t``
(:mod:`quant_research_engine.guards.poison`), and requires bit-identical decisions at every
replayed instant. The real world is replayed once (to the last sampled instant) and compared
with every poisoned replay.

Isolation is in-process. Before each world the store registry is pointed at that world's store
(``open_store`` with any path returns it), and every module that is not standard library, not a
third-party package, and not part of this package is removed from ``sys.modules`` together with
the subject's own module; the subject is then imported again and instantiated from its spec.
A handle opened at import, in a constructor, by path, or memoized in a helper module
(``functools.lru_cache`` or a module global) therefore comes from the world being replayed.
Limits: a module imported before the audit started that lives outside the subjects' directories
survives the purge, and state planted inside this package's own modules is not reset.
"""

from __future__ import annotations

import importlib
import os
import sys
import sysconfig
import types
from dataclasses import dataclass, field

from quant_research_engine.backtest import RunConfig, run_strategy
from quant_research_engine.context.decision import decision_instruments, decision_key
from quant_research_engine.store import registry

PACKAGE = "quant_research_engine"


def _site_dirs() -> list[str]:
    paths = sysconfig.get_paths()
    dirs = {
        os.path.normcase(os.path.abspath(paths[k]))
        for k in ("purelib", "platlib", "stdlib", "platstdlib")
        if k in paths
    }
    dirs.add(os.path.normcase(os.path.abspath(sys.base_prefix)))
    return sorted(dirs)


_SITE = _site_dirs()


def _protected(name: str, mod) -> bool:
    top = name.split(".")[0]
    if top in sys.stdlib_module_names or top in sys.builtin_module_names:
        return True
    if top == PACKAGE:
        return True
    f = getattr(mod, "__file__", None)
    if not f:
        return True
    f = os.path.normcase(os.path.abspath(f))
    return any(f.startswith(d + os.sep) or f == d for d in _SITE)


def purge_modules(subject_module: str, roots: list[str], baseline: set[str]) -> list[str]:
    """Remove the subject module, every non-protected module imported after ``baseline``, and
    every non-protected module whose file lies under one of ``roots``."""
    roots_n = [os.path.normcase(os.path.abspath(r)) for r in roots]
    gone = []
    for name in list(sys.modules):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        if name == subject_module or name.startswith(subject_module + "."):
            gone.append(name)
            continue
        if _protected(name, mod):
            continue
        f = os.path.normcase(os.path.abspath(getattr(mod, "__file__", "") or ""))
        if name not in baseline or any(f.startswith(r + os.sep) for r in roots_n):
            gone.append(name)
    for name in gone:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return gone


@dataclass
class Subject:
    """An audited strategy: ``spec`` = (module, class, params)."""

    id: str
    name: str
    spec: tuple
    kind: str = "strategy"  # strategy | feature | run | cv
    named_guard: str = ""
    roots: list[str] = field(default_factory=list)
    run_overrides: dict = field(default_factory=dict)
    feature: object | None = None
    validation: dict | None = None


# ---------------------------------------------------------------------- package state


def _fingerprint(v):
    """Identity of a value, plus the identities of a container's items (shallow)."""
    if isinstance(v, list | tuple):
        return (id(v), tuple(id(x) for x in v))
    if isinstance(v, dict):
        return (id(v), tuple((k, id(x)) for k, x in v.items()))
    if isinstance(v, set | frozenset):
        return (id(v), len(v))
    return id(v)


def _dunder(k: str) -> bool:
    return k.startswith("__") and k.endswith("__")


def _tracked(k: str, v) -> bool:
    """Module globals worth comparing: not dunders and not submodule bindings (the import
    system adds and rebinds those when a module is first imported or re-imported)."""
    return not _dunder(k) and not isinstance(v, types.ModuleType)


def _is_model(cls) -> bool:
    try:
        from pydantic import BaseModel

        return issubclass(cls, BaseModel)
    except TypeError:
        return False


def package_snapshot(skip: str) -> dict:
    """Module globals and class attributes of every loaded module of this package (except the
    subject's own module), with the original objects so that they can be restored."""
    snap = {}
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == PACKAGE or name.startswith(PACKAGE + ".")):
            continue
        if name == skip or name.startswith(skip + "."):
            continue
        g = {k: v for k, v in vars(mod).items() if _tracked(k, v)}
        snap[("m", name)] = (mod, g, {k: _fingerprint(v) for k, v in g.items()})
        for v in g.values():
            if (
                isinstance(v, type)
                and getattr(v, "__module__", "").startswith(PACKAGE)
                and not _is_model(v)
                and ("c", id(v)) not in snap
            ):
                cd = {k: x for k, x in v.__dict__.items() if not _dunder(k)}
                snap[("c", id(v))] = (v, cd, {k: _fingerprint(x) for k, x in cd.items()})
    return snap


def restore_package(snap: dict) -> list[str]:
    """Compare with the snapshot, undo every change, and return what had changed."""
    changed = []
    for key, (obj, orig, fp) in snap.items():
        src = vars(obj) if key[0] == "m" else obj.__dict__
        cur = {k: v for k, v in src.items() if _tracked(k, v)}
        for k in sorted(set(cur) | set(orig)):
            if k not in orig or k not in cur or _fingerprint(cur[k]) != fp[k]:
                label = obj.__name__ if key[0] == "m" else f"{obj.__module__}.{obj.__qualname__}"
                changed.append(f"{label}.{k}")
                try:
                    if k in orig:
                        setattr(obj, k, orig[k])
                    else:
                        delattr(obj, k)
                except (AttributeError, TypeError):
                    pass
    return changed


def freeze_store(store) -> None:
    """Make a world's tables and panel read-only, so a subject cannot corrupt shared worlds."""
    for t in store.tables.values():
        for arr in t.columns.values():
            arr.setflags(write=False)
    p = store.panel()
    for v in vars(p).values():
        if hasattr(v, "setflags"):
            v.setflags(write=False)


@dataclass
class WorldReplay:
    keys: dict[int, tuple]
    g7_violations: list[tuple]
    error: str | None = None
    error_k: int | None = None
    state_changes: list = field(default_factory=list)
    unpurged: int = 0


def replay_world(
    subject: Subject, store, rc: RunConfig, stop_k: int, baseline: set[str]
) -> WorldReplay:
    """Replay up to ``stop_k``; decisions made before an exception are kept, with the bar of the
    error. Package state changed by the subject is reported and undone."""
    from quant_research_engine.inference import splitters

    decisions: list = []
    progress = {"k": -1}
    error = None
    snap = package_snapshot(subject.spec[0])
    with registry.world(store), splitters.observe() as unpurged:
        purge_modules(subject.spec[0], subject.roots, baseline)
        try:
            mod = importlib.import_module(subject.spec[0])
            strat = getattr(mod, subject.spec[1])(**dict(subject.spec[2] or {}))
            run_strategy(store, strat, rc, stop_after=stop_k, record=decisions, progress=progress)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
    changes = restore_package(snap)
    keys = {}
    g7 = []
    for k, d, uni in decisions:
        keys[k] = decision_key(d)
        outside = sorted(set(decision_instruments(d)) - set(uni))
        if outside:
            g7.append((k, outside))
    err_k = max(progress["k"], 0) if error is not None else None
    return WorldReplay(keys, g7, error, err_k, changes, len(unpurged))


@dataclass
class AuditResult:
    subject: str
    instants: list[int]
    caught: list[bool]
    first_diff: list[int | None]
    g7_violations: int
    errors: list[str]
    state_changes: list = field(default_factory=list)
    unpurged_splits: int = 0
    real_error: str | None = None

    @property
    def any_caught(self) -> bool:
        return any(self.caught)


def invariance_audit(
    subject: Subject, store, instants: list[int], rc: RunConfig, poisoned: dict[int, object]
) -> AuditResult:
    """``poisoned[k]`` is the store poisoned after the close of calendar bar ``k``.

    At instant ``k`` the two worlds differ if a decision at or before ``k`` differs, if one world
    raises at or before ``k`` and the other does not, if they raise at different bars, or if the
    subject changed this package's state (a channel the purge does not reset)."""
    baseline = set(sys.modules)
    instants = sorted(int(k) for k in instants)
    freeze_store(store)
    real = replay_world(subject, store, rc, max(instants), baseline)
    caught, first, errors = [], [], []
    changes = list(real.state_changes)
    unpurged = real.unpurged
    if real.error:
        errors.append(f"real@{real.error_k}: {real.error}")
    for k in instants:
        freeze_store(poisoned[k])
        w = replay_world(subject, poisoned[k], rc, k, baseline)
        changes += w.state_changes
        unpurged += w.unpurged
        r_err = real.error_k if real.error is not None and real.error_k <= k else None
        diff = None
        if (r_err is None) != (w.error is None) or (r_err is not None and r_err != w.error_k):
            diff = min(x for x in (r_err, w.error_k) if x is not None)
        lim = k if diff is None else diff
        for kk in sorted({x for x in w.keys if x <= lim} | {x for x in real.keys if x <= lim}):
            if w.keys.get(kk) != real.keys.get(kk):
                diff = kk if diff is None else min(diff, kk)
                break
        if real.state_changes or w.state_changes:
            diff = k if diff is None else diff
        if w.error:
            errors.append(f"poisoned@{k} (bar {w.error_k}): {w.error}")
        caught.append(diff is not None)
        first.append(diff)
    return AuditResult(
        subject.id,
        instants,
        caught,
        first,
        len(real.g7_violations),
        errors,
        sorted(set(changes)),
        unpurged,
        real.error,
    )
