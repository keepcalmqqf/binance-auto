from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


class Signal(str, Enum):
    HOLD = "HOLD"
    BUY = "BUY"
    SELL = "SELL"


class Mode(str, Enum):
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


@dataclass(frozen=True)
class Candle:
    open_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: int

    @classmethod
    def from_binance(cls, row: list[Any]) -> Candle:
        return cls(
            open_time=int(row[0]),
            open=Decimal(str(row[1])),
            high=Decimal(str(row[2])),
            low=Decimal(str(row[3])),
            close=Decimal(str(row[4])),
            volume=Decimal(str(row[5])),
            close_time=int(row[6]),
        )


@dataclass(frozen=True)
class Fill:
    price: Decimal
    quantity: Decimal
    commission: Decimal
    commission_asset: str
    trade_id: str

    @classmethod
    def from_binance(cls, row: dict[str, Any]) -> Fill:
        return cls(
            price=Decimal(str(row.get("price", "0"))),
            quantity=Decimal(str(row.get("qty", "0"))),
            commission=Decimal(str(row.get("commission", "0"))),
            commission_asset=str(row.get("commissionAsset", "")),
            trade_id=str(row.get("tradeId", "")),
        )


@dataclass(frozen=True)
class OrderResult:
    client_order_id: str
    order_id: str
    symbol: str
    side: Signal
    status: str
    executed_quantity: Decimal
    quote_quantity: Decimal
    transact_time: int
    fills: tuple[Fill, ...] = ()

    @classmethod
    def from_binance(cls, data: dict[str, Any]) -> OrderResult:
        return cls(
            client_order_id=str(data.get("clientOrderId", "")),
            order_id=str(data.get("orderId", "")),
            symbol=str(data.get("symbol", "")),
            side=Signal(str(data.get("side", "HOLD"))),
            status=str(data.get("status", "UNKNOWN")),
            executed_quantity=Decimal(str(data.get("executedQty", "0"))),
            quote_quantity=Decimal(str(data.get("cummulativeQuoteQty", "0"))),
            transact_time=int(
                data.get("transactTime", data.get("updateTime", data.get("time", 0)))
            ),
            fills=tuple(Fill.from_binance(fill) for fill in data.get("fills", [])),
        )
