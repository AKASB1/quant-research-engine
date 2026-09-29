"""Small event-driven buy/sell step."""
from quant_research_engine.portfolio import Portfolio
from quant_research_engine.costs import execution_cost

def trade(portfolio: Portfolio, price: float, units: float, fee: float = 0.0) -> None:
    if price <= 0 or fee < 0:
        raise ValueError("invalid price or fee")
    debit = price * units + (execution_cost(price * units, fee) if units else 0.0)
    if debit > portfolio.cash:
        raise ValueError("insufficient cash")
    if portfolio.units + units < 0:
        raise ValueError("insufficient units")
    portfolio.cash -= debit
    portfolio.units += units
