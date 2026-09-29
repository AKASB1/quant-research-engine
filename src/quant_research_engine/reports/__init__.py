"""Minimal summary record."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Summary:
    total_return: float
    max_drawdown: float
