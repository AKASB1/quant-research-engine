"""Past-only simple moving average."""
from quant_research_engine.data import Bar

def moving_average(bars: list[Bar], window: int) -> float:
    if window < 1 or len(bars) < window:
        raise ValueError("insufficient history or invalid window")
    return sum(bar.close for bar in bars[-window:]) / window
