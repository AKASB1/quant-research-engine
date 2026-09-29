"""Fixed plus proportional execution cost."""
def execution_cost(notional: float, fixed: float = 0.0, rate: float = 0.0) -> float:
    if min(notional, fixed, rate) < 0:
        raise ValueError("cost inputs must be nonnegative")
    return fixed + abs(notional) * rate
