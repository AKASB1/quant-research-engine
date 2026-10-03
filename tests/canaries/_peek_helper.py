"""Canary helper: a store handle opened on first use and memoized with lru_cache.

Re-importing a canary module alone leaves this cached handle in place, so an audit that does
not purge helper modules would read the first world's store in every world.
"""

import functools
import os

from quant_research_engine.store.registry import open_store


@functools.lru_cache(maxsize=None)  # noqa: UP033 (the TASK names lru_cache)
def handle():
    return open_store(os.environ.get("QRE_STORE"))
