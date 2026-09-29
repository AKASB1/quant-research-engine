import unittest
from datetime import datetime, timedelta, timezone
from quant_research_engine.data import Bar
from quant_research_engine.signals import trend_signal
from quant_research_engine.portfolio import Portfolio
from quant_research_engine.backtest import trade

class ResearchTests(unittest.TestCase):
    def test_availability_and_trade(self):
        now = datetime.now(timezone.utc)
        bars = [Bar("X", now, now, 100), Bar("X", now + timedelta(days=1), now + timedelta(days=2), 110)]
        self.assertEqual(trend_signal(bars, 1, now), 0)
        portfolio = Portfolio(1000)
        trade(portfolio, 100, 2, 1)
        self.assertEqual(portfolio.value(100), 999)
