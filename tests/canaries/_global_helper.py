"""Canary helper: a store handle kept in a module-level global after first use."""

import os

from quant_research_engine.store.registry import open_store

_HANDLE = None


def handle():
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = open_store(os.environ.get("QRE_STORE"))
    return _HANDLE
