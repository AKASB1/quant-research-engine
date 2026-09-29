"""Simple trend signal based on available history."""
from quant_research_engine.data import Bar
from quant_research_engine.features import moving_average

def trend_signal(bars: list[Bar], window: int, as_of) -> int:
    known = [bar for bar in bars if bar.available_at <= as_of]
    if len(known) < window:
        return 0
    return int(known[-1].close > moving_average(known, window))
