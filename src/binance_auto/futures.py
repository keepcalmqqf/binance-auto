from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .binance import (
    BinanceAPIError,
    BinanceClient,
    BinanceError,
    BinanceTransportError,
)
from .models import Candle, OrderResult, Signal
from .rules import SymbolRules, decimal_to_string


@dataclass(frozen=True)
class FuturesAccountState:
    available_balance: Decimal
    wallet_balance: Decimal
    margin_balance: Decimal
    position_quantity: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal
    leverage: int
    margin_type: str

    @classmethod
    def from_account(cls, payload: dict[str, Any], symbol: str) -> FuturesAccountState:
        positions = [
            item
            for item in payload.get("positions", [])
            if item.get("symbol") == symbol
        ]
        position = next(
            (item for item in positions if item.get("positionSide", "BOTH") == "BOTH"),
            positions[0] if positions else {},
        )
        return cls(
            available_balance=Decimal(str(payload.get("availableBalance", "0"))),
            wallet_balance=Decimal(str(payload.get("totalWalletBalance", "0"))),
            margin_balance=Decimal(str(payload.get("totalMarginBalance", "0"))),
            position_quantity=Decimal(str(position.get("positionAmt", "0"))),
            entry_price=Decimal(str(position.get("entryPrice", "0"))),
            unrealized_pnl=Decimal(str(position.get("unrealizedProfit", "0"))),
            leverage=int(position.get("leverage", 0)),
            margin_type=str(
                position.get(
                    "marginType", "ISOLATED" if position.get("isolated") else "CROSSED"
                )
            ).upper(),
        )


class BinanceFuturesClient(BinanceClient):
    def synchronize_time(self) -> int:
        before = int(time.time() * 1000)
        payload = self._request("GET", "/fapi/v1/time")
        after = int(time.time() * 1000)
        server_time = int(payload["serverTime"])
        self._server_offset_ms = server_time - (before + after) // 2
        return server_time

    def get_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        payload = self._request("GET", "/fapi/v1/klines", params)
        return [Candle.from_binance(row) for row in payload]

    def get_symbol_rules(self, symbol: str) -> SymbolRules:
        payload = self._request("GET", "/fapi/v1/exchangeInfo", {"symbol": symbol})
        return SymbolRules.from_futures_exchange_info(payload, symbol)

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_account_state(self, symbol: str) -> FuturesAccountState:
        return FuturesAccountState.from_account(self.get_account(), symbol)

    def get_position_mode(self) -> bool:
        payload = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
        return str(payload.get("dualSidePosition", "false")).lower() == "true"

    def set_margin_type(self, symbol: str, margin_type: str) -> None:
        try:
            self._request(
                "POST",
                "/fapi/v1/marginType",
                {"symbol": symbol, "marginType": margin_type},
                signed=True,
                allow_retries=False,
            )
        except BinanceAPIError as exc:
            if exc.code != -4046:
                raise

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": leverage},
            signed=True,
            allow_retries=False,
        )

    def configure_trading(self, symbol: str, leverage: int, margin_type: str) -> None:
        if self.get_position_mode():
            raise BinanceError(
                "Hedge mode is not supported; switch to One-way mode manually before running"
            )
        self.set_margin_type(symbol, margin_type)
        self.set_leverage(symbol, leverage)

    def query_order(self, symbol: str, client_order_id: str) -> OrderResult:
        payload = self._request(
            "GET",
            "/fapi/v1/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
            signed=True,
        )
        return OrderResult.from_binance(payload)

    def place_market_order(
        self,
        symbol: str,
        side: Signal,
        client_order_id: str,
        *,
        quantity: Decimal | None = None,
        quote_order_quantity: Decimal | None = None,
        reduce_only: bool = False,
    ) -> OrderResult:
        if side not in (Signal.BUY, Signal.SELL):
            raise ValueError("market orders require BUY or SELL")
        if quantity is None or quantity <= 0 or quote_order_quantity is not None:
            raise ValueError("futures market orders require a positive base quantity")
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "type": "MARKET",
            "quantity": decimal_to_string(quantity),
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        try:
            payload = self._request(
                "POST", "/fapi/v1/order", params, signed=True, allow_retries=False
            )
        except BinanceTransportError:
            return self._reconcile_order(symbol, client_order_id)
        except BinanceAPIError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                return self._reconcile_order(symbol, client_order_id)
            raise
        return OrderResult.from_binance(payload)
