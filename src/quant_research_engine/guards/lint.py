"""Guard G3: a static allow-list lint of strategy and feature modules (a lint, not a sandbox).

An import passes only if it comes from ``__future__``, ``math``, ``typing``, ``dataclasses``,
``functools``, ``collections``, ``itertools``, NumPy, SciPy (not ``scipy.io``), or the public
strategy API of this package (the context, the decision types, the features, the signals, the
portfolio rules, and the strategy base). ``open``, ``eval``, ``exec``, ``__import__``,
``importlib``, ``scipy.io``, and NumPy's file readers (``load``, ``loadtxt``, ``genfromtxt``,
``fromfile``, ``memmap``) are rejected wherever they appear, and so are reflective builtins
(``getattr``, ``setattr``, ``delattr``, ``globals``, ``locals``, ``vars``), dunder attribute
access other than ``__init__``, and ``import quant_research_engine...`` (it binds the package
root, from which internal modules are reachable by attribute; ``from ... import`` is required).
A determined author can get around a lint, which is why the replay audit (G2) exists.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

ALLOWED_TOP = {
    "__future__",
    "math",
    "typing",
    "dataclasses",
    "functools",
    "collections",
    "itertools",
    "numpy",
    "scipy",
}
ALLOWED_PACKAGE = (
    "quant_research_engine.context",
    "quant_research_engine.features",
    "quant_research_engine.signals",
    "quant_research_engine.portfolio",
    "quant_research_engine.strategies.base",
)
BANNED_NAMES = {
    "open",
    "eval",
    "exec",
    "__import__",
    "compile",
    "importlib",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "input",
}
ALLOWED_DUNDERS = {"__init__", "__future__"}
# attribute names that only ever mean the dangerous builtins or modules (``panel.open`` is data)
ATTR_BANNED = {"eval", "exec", "importlib", "getattr", "setattr", "delattr", "globals", "locals"}
NUMPY_READERS = {"load", "loadtxt", "genfromtxt", "fromfile", "memmap"}


@dataclass(frozen=True)
class LintIssue:
    line: int
    message: str


def _module_allowed(name: str) -> bool:
    if name == "scipy.io" or name.startswith("scipy.io."):
        return False
    top = name.split(".")[0]
    if top in ALLOWED_TOP:
        return True
    return any(name == p or name.startswith(p + ".") for p in ALLOWED_PACKAGE)


def lint_source(source: str, filename: str = "<strategy>") -> list[LintIssue]:
    tree = ast.parse(source, filename=filename)
    issues: list[LintIssue] = []
    numpy_aliases = {"numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "numpy" and a.asname:
                    numpy_aliases.add(a.asname)
                if a.name.split(".")[0] == "quant_research_engine":
                    # a plain import binds the package root, from which every internal module
                    # (the store among them) is reachable by attribute; use "from ... import"
                    issues.append(
                        LintIssue(
                            node.lineno,
                            f"use 'from {a.name} import ...' instead of 'import {a.name}'",
                        )
                    )
                elif not _module_allowed(a.name):
                    issues.append(LintIssue(node.lineno, f"import of {a.name!r} is not allowed"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                issues.append(LintIssue(node.lineno, "relative imports are not allowed"))
                continue
            if not _module_allowed(mod):
                issues.append(LintIssue(node.lineno, f"import from {mod!r} is not allowed"))
            for a in node.names:
                full = f"{mod}.{a.name}"
                if mod.split(".")[0] == "numpy" and a.name in NUMPY_READERS:
                    issues.append(LintIssue(node.lineno, f"numpy file reader {a.name!r}"))
                if full == "scipy.io" or a.name in BANNED_NAMES:
                    issues.append(LintIssue(node.lineno, f"{full!r} is not allowed"))
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            issues.append(LintIssue(node.lineno, f"use of {node.id!r} is not allowed"))
        elif (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr not in ALLOWED_DUNDERS
        ):
            issues.append(LintIssue(node.lineno, f"dunder attribute {node.attr!r} is not allowed"))
        elif isinstance(node, ast.Attribute):
            if (
                node.attr in NUMPY_READERS
                and isinstance(node.value, ast.Name)
                and node.value.id in numpy_aliases
            ):
                issues.append(LintIssue(node.lineno, f"numpy file reader {node.attr!r}"))
            if node.attr == "io" and isinstance(node.value, ast.Name) and node.value.id == "scipy":
                issues.append(LintIssue(node.lineno, "scipy.io is not allowed"))
            if node.attr in ATTR_BANNED:
                issues.append(LintIssue(node.lineno, f"use of {node.attr!r} is not allowed"))
    return sorted(set(issues), key=lambda x: (x.line, x.message))


def lint_file(path: str) -> list[LintIssue]:
    with open(path, encoding="utf-8") as f:
        return lint_source(f.read(), path)
