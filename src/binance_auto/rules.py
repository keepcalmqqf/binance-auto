from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any


class RuleViolation(ValueError):
    pass


def decimal_to_string(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def floor_to_precision(value: Decimal, precision: int) -> Decimal:
    quantum = Decimal(1).scaleb(-precision)
    return value.quantize(quantum, rounding=ROUND_DOWN)


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    base_asset: str
    quote_asset: str
    base_precision: int
    quote_precision: int
    tick_size: Decimal
    min_quantity: Decimal
    max_quantity: Decimal
    step_size: Decimal
    min_notional: Decimal
    max_notional: Decimal

    @classmethod
    def from_exchange_info(cls, payload: dict[str, Any], symbol: str) -> SymbolRules:
        symbols = payload.get("symbols", [])
        info = next((item for item in symbols if item.get("symbol") == symbol), None)
        if info is None:
            raise RuleViolation(f"symbol {symbol} was not returned by exchangeInfo")
        if info.get("status") != "TRADING" or not info.get("isSpotTradingAllowed", False):
            raise RuleViolation(f"symbol {symbol} is not available for Spot trading")

        filters = {item.get("filterType"): item for item in info.get("filters", [])}
        lot = filters.get("LOT_SIZE", {})
        market_lot = filters.get("MARKET_LOT_SIZE", {})
        if Decimal(str(market_lot.get("stepSize", "0"))) > 0:
            lot = market_lot
        price_filter = filters.get("PRICE_FILTER", {})
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        return cls(
            symbol=symbol,
            base_asset=str(info["baseAsset"]),
            quote_asset=str(info["quoteAsset"]),
            base_precision=int(info.get("baseAssetPrecision", 8)),
            quote_precision=int(info.get("quoteAssetPrecision", 8)),
            tick_size=Decimal(str(price_filter.get("tickSize", "0"))),
            min_quantity=Decimal(str(lot.get("minQty", "0"))),
            max_quantity=Decimal(str(lot.get("maxQty", "0"))),
            step_size=Decimal(str(lot.get("stepSize", "0"))),
            min_notional=Decimal(str(notional.get("minNotional", "0"))),
            max_notional=Decimal(str(notional.get("maxNotional", "0"))),
        )

    @classmethod
    def from_futures_exchange_info(
        cls, payload: dict[str, Any], symbol: str
    ) -> SymbolRules:
        symbols = payload.get("symbols", [])
        info = next((item for item in symbols if item.get("symbol") == symbol), None)
        if info is None:
            raise RuleViolation(f"contract {symbol} was not returned by exchangeInfo")
        if info.get("status") != "TRADING" or info.get("contractType") != "PERPETUAL":
            raise RuleViolation(f"contract {symbol} is not a trading USDT-M perpetual")

        filters = {item.get("filterType"): item for item in info.get("filters", [])}
        lot = filters.get("LOT_SIZE", {})
        market_lot = filters.get("MARKET_LOT_SIZE", {})
        if Decimal(str(market_lot.get("stepSize", "0"))) > 0:
            lot = market_lot
        price_filter = filters.get("PRICE_FILTER", {})
        notional = filters.get("MIN_NOTIONAL", {})
        return cls(
            symbol=symbol,
            base_asset=str(info["baseAsset"]),
            quote_asset=str(info["quoteAsset"]),
            base_precision=int(info.get("quantityPrecision", 8)),
            quote_precision=int(info.get("quotePrecision", 8)),
            tick_size=Decimal(str(price_filter.get("tickSize", "0"))),
            min_quantity=Decimal(str(lot.get("minQty", "0"))),
            max_quantity=Decimal(str(lot.get("maxQty", "0"))),
            step_size=Decimal(str(lot.get("stepSize", "0"))),
            min_notional=Decimal(
                str(notional.get("minNotional", notional.get("notional", "0")))
            ),
            max_notional=Decimal(str(notional.get("maxNotional", "0"))),
        )

    def floor_quantity(self, quantity: Decimal) -> Decimal:
        return floor_to_step(quantity, self.step_size)

    def floor_quote(self, quantity: Decimal) -> Decimal:
        return floor_to_precision(quantity, self.quote_precision)

    def validate_notional(self, notional: Decimal) -> None:
        if self.min_notional > 0 and notional < self.min_notional:
            raise RuleViolation(
                f"order notional {decimal_to_string(notional)} is below minimum "
                f"{decimal_to_string(self.min_notional)} {self.quote_asset}"
            )
        if self.max_notional > 0 and notional > self.max_notional:
            raise RuleViolation(
                f"order notional {decimal_to_string(notional)} exceeds maximum "
                f"{decimal_to_string(self.max_notional)} {self.quote_asset}"
            )

    def validate_market_buy(self, quote_quantity: Decimal, reference_price: Decimal) -> None:
        if quote_quantity <= 0:
            raise RuleViolation("buy quote quantity must be positive")
        self.validate_notional(quote_quantity)
        if reference_price > 0:
            estimated_quantity = quote_quantity / reference_price
            if self.min_quantity > 0 and estimated_quantity < self.min_quantity:
                raise RuleViolation("estimated buy quantity is below the symbol minimum")
            if self.max_quantity > 0 and estimated_quantity > self.max_quantity:
                raise RuleViolation("estimated buy quantity exceeds the symbol maximum")

    def validate_quantity(self, quantity: Decimal, reference_price: Decimal) -> None:
        if quantity <= 0:
            raise RuleViolation("order quantity must be positive")
        if self.min_quantity > 0 and quantity < self.min_quantity:
            raise RuleViolation(
                f"order quantity is below minimum {decimal_to_string(self.min_quantity)}"
            )
        if self.max_quantity > 0 and quantity > self.max_quantity:
            raise RuleViolation(
                f"order quantity exceeds maximum {decimal_to_string(self.max_quantity)}"
            )
        if self.step_size > 0 and self.floor_quantity(quantity) != quantity:
            raise RuleViolation("order quantity is not aligned to the symbol step size")
        self.validate_notional(quantity * reference_price)

    def validate_market_sell(self, quantity: Decimal, reference_price: Decimal) -> None:
        self.validate_quantity(quantity, reference_price)
