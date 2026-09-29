"""Return and drawdown helpers."""
def simple_return(start: float, end: float) -> float:
    if start <= 0:
        raise ValueError("start must be positive")
    return end / start - 1

def max_drawdown(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("positive values required")
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, 1 - value / peak)
    return drawdown
