"""Per-component random streams (the stream derivation of the shared contract, section 1.2).

    c  = fnv1a64(name)                       64-bit FNV-1a over the UTF-8 bytes
    s1 = splitmix64(seed XOR c)
    s2 = splitmix64(s1 XOR 0x9E3779B97F4A7C15 XOR c)
    stream = NumPy PCG64DXSM, 128-bit state (s1 << 64) | s2, fixed odd increment

Every component draws from its own stream, so adding or removing a component never shifts
another component's numbers. The ``random`` module, ``hash()``, and the global NumPy state
never reach a result. Adapted from the owner's rollout-engine (MIT).
"""

from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
PCG_INC = ((6364136223846793005 << 64) | 1442695040888963407) | 1


def fnv1a64(text: str) -> int:
    """64-bit FNV-1a over the UTF-8 bytes of ``text``."""
    h = FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * FNV_PRIME) & MASK64
    return h


def splitmix64(x: int) -> int:
    """Add the golden constant to ``x`` (mod 2^64) and apply the SplitMix64 finalizer."""
    x = (x + GOLDEN) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def stream_seeds(seed: int, name: str) -> tuple[int, int]:
    c = fnv1a64(name)
    s1 = splitmix64((seed & MASK64) ^ c)
    s2 = splitmix64(s1 ^ GOLDEN ^ c)
    return s1, s2


def stream(seed: int, name: str) -> np.random.Generator:
    """A fresh, deterministic generator for component ``name`` of run ``seed``."""
    s1, s2 = stream_seeds(seed, name)
    bg = np.random.PCG64DXSM()
    bg.state = {
        "bit_generator": "PCG64DXSM",
        "state": {"state": (s1 << 64) | s2, "inc": PCG_INC},
        "has_uint32": 0,
        "uinteger": 0,
    }
    return np.random.Generator(bg)


def derived_seed(seed: int, name: str) -> int:
    """A 63-bit seed derived from (seed, name), for components that take an integer seed."""
    return int(stream(seed, name).integers(0, 2**63 - 1, dtype=np.int64))
