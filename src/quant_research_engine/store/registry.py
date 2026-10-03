"""Store registry: the one way code opens a store by path.

``open_store(path)`` reads the store at ``path``, unless the replay audit has installed a world:
then every ``open_store`` call, with any path, returns that world's store. This is what makes a
module-level handle, a handle opened in a constructor, and a read by path see the poisoned
world during the audit (guard G2). Stores are not memoized here.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from quant_research_engine.store.store import Store, read_store

_world: Store | None = None
ENV_STORE = "QRE_STORE"


def open_store(path: str | None = None) -> Store:
    if _world is not None:
        return _world
    if path is None:
        path = os.environ.get(ENV_STORE)
        if not path:
            raise ValueError(f"no store path given and {ENV_STORE} is not set")
    return read_store(path)


def current_world() -> Store | None:
    return _world


@contextmanager
def world(store: Store):
    """Within the block, ``open_store`` with any path returns ``store``."""
    global _world
    prev = _world
    _world = store
    try:
        yield store
    finally:
        _world = prev
