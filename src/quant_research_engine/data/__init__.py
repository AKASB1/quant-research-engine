"""Point-in-time bar record."""
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    available_at: datetime
    close: float

    def __post_init__(self) -> None:
        if self.available_at < self.timestamp or self.close <= 0:
            raise ValueError("invalid availability or close price")
