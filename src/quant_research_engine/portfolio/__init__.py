"""Single-asset cash and position accounting."""
from dataclasses import dataclass

@dataclass
class Portfolio:
    cash: float
    units: float = 0.0

    def value(self, price: float) -> float:
        return self.cash + self.units * price
