"""Loading audit subjects from a directory of canary modules.

Each canary module (a file not starting with ``_`` or ``test``) defines a dict ``CANARY`` with
``id``, ``name``, ``guard`` (the guard that must catch it), ``class``, and optionally ``params``,
``kind`` (strategy, feature, run, cv), and ``run`` (run-configuration overrides). The loader puts
the directory on ``sys.path``, reads the metadata, and removes the modules again from
``sys.modules`` so that the audit imports them fresh in every world.
"""

from __future__ import annotations

import importlib
import os
import sys

from quant_research_engine.guards.replay import Subject, _protected


def load_subjects(directory: str) -> list[Subject]:
    directory = os.path.abspath(directory)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    out = []
    names = sorted(
        f[:-3]
        for f in os.listdir(directory)
        if f.endswith(".py") and not f.startswith(("_", "test"))
    )
    for name in names:
        before = set(sys.modules)
        mod = importlib.import_module(name)
        meta = dict(mod.CANARY)
        out.append(
            Subject(
                id=meta["id"],
                name=meta["name"],
                spec=(meta.get("module", name), meta["class"], dict(meta.get("params", {}))),
                kind=meta.get("kind", "strategy"),
                named_guard=meta["guard"],
                roots=[directory],
                run_overrides=dict(meta.get("run", {})),
                feature=getattr(mod, meta["feature"]) if "feature" in meta else None,
                validation=meta.get("validation"),
            )
        )
        for m in sorted(set(sys.modules) - before):  # only canary and user modules, never libraries
            mod = sys.modules.get(m)
            if mod is not None and not _protected(m, mod):
                sys.modules.pop(m, None)
        sys.modules.pop(name, None)
    return out
