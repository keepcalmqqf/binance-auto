from __future__ import annotations

import hashlib
import hmac
import time
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import httpx

from .models import Candle, OrderResult, Signal
from .rules import SymbolRules, decimal_to_string


class BinanceError(RuntimeError):
    pass


class BinanceTransportError(BinanceError):
    pass


class BinanceAPIError(BinanceError):
    def __init__(self, status_code: int, code: int | None, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        code_text = "unknown" if code is None else str(code)
        super().__init__(f"Binance API error {code_text} (HTTP {status_code}): {message}")


class OrderUncertainError(BinanceError):
    def __init__(self, client_order_id: str):
        self.client_order_id = client_order_id
        super().__init__(
            f"order outcome is uncertain for clientOrderId={client_order_id}; "
            "reconcile it on Binance before restarting this signal"
        )


class BinanceClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout_seconds: float = 10.0,
        recv_window: int = 5000,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self._api_secret = api_secret
        self.recv_window = recv_window
        self.max_retries = max_retries
        self._server_offset_ms: int | None = None
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": "binance-auto/0.1.0", "Accept": "application/json"},
        )

    def __enter__(self) -> BinanceClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def now_ms(self) -> int:
        return int(time.time() * 1000) + (self._server_offset_ms or 0)

    def synchronize_time(self) -> int:
        before = int(time.time() * 1000)
        payload = self._request("GET", "/api/v3/time")
        after = int(time.time() * 1000)
        server_time = int(payload["serverTime"])
        self._server_offset_ms = server_time - (before + after) // 2
        return server_time

    def _credentials(self) -> tuple[str, str]:
        if not self.api_key or not self._api_secret:
            raise BinanceError("API credentials are required for private endpoints")
        return self.api_key, self._api_secret

    @staticmethod
    def signature(secret: str, query_string: str) -> str:
        return hmac.new(
            secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _signed_params(self, params: dict[str, Any]) -> dict[str, Any]:
        self._credentials()
        if self._server_offset_ms is None:
            self.synchronize_time()
        signed = dict(params)
        signed.setdefault("timestamp", self.now_ms())
        signed.setdefault("recvWindow", self.recv_window)
        query = urlencode(signed)
        signed["signature"] = self.signature(self._api_secret or "", query)
        return signed

    @staticmethod
    def _error(response: httpx.Response) -> BinanceAPIError:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        code = payload.get("code") if isinstance(payload, dict) else None
        message = (
            str(payload.get("msg", "request failed"))
            if isinstance(payload, dict)
            else "request failed"
        )
        return BinanceAPIError(response.status_code, code, message)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = False,
        allow_retries: bool = True,
    ) -> Any:
        original = dict(params or {})
        retries = self.max_retries if allow_retries else 0
        for attempt in range(retries + 1):
            try:
                request_params = self._signed_params(original) if signed else dict(original)
                headers = {"X-MBX-APIKEY": self.api_key} if signed else None
                response = self._http.request(
                    method,
                    path,
                    params=request_params if method == "GET" else None,
                    data=request_params if method != "GET" else None,
                    headers=headers,
                )
            except httpx.RequestError as exc:
                if allow_retries and attempt < retries:
                    time.sleep(min(0.5 * (2**attempt), 5.0))
                    continue
                raise BinanceTransportError(
                    f"{method} {path} failed due to a network or timeout error"
                ) from exc

            if response.is_success:
                try:
                    return response.json()
                except ValueError as exc:
                    raise BinanceError(f"{method} {path} returned invalid JSON") from exc

            error = self._error(response)
            retryable = response.status_code == 429 or response.status_code >= 500
            if allow_retries and retryable and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 0.5 * (2**attempt)
                except ValueError:
                    delay = 0.5 * (2**attempt)
                time.sleep(min(max(delay, 0.1), 10.0))
                continue
            raise error
        raise BinanceError("unreachable request retry state")

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
        payload = self._request("GET", "/api/v3/klines", params)
        return [Candle.from_binance(row) for row in payload]

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: int,
        end_time: int,
    ) -> list[Candle]:
        candles: dict[int, Candle] = {}
        cursor = start_time
        page_size = 1000
        while cursor <= end_time:
            batch = self.get_klines(
                symbol,
                interval,
                start_time=cursor,
                end_time=end_time,
                limit=page_size,
            )
            if not batch:
                break
            for candle in batch:
                candles[candle.open_time] = candle
            next_cursor = batch[-1].open_time + 1
            if len(batch) < page_size or next_cursor <= cursor:
                break
            cursor = next_cursor
        return [candles[key] for key in sorted(candles)]

    def get_recent_closed_klines(
        self, symbol: str, interval: str, *, limit: int
    ) -> list[Candle]:
        if self._server_offset_ms is None:
            self.synchronize_time()
        candles = self.get_klines(symbol, interval, limit=limit)
        return [candle for candle in candles if candle.close_time < self.now_ms()]

    def get_symbol_rules(self, symbol: str) -> SymbolRules:
        payload = self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol})
        return SymbolRules.from_exchange_info(payload, symbol)

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", "/api/v3/account", signed=True)

    @staticmethod
    def free_balance(account: dict[str, Any], asset: str) -> Decimal:
        balance = next(
            (item for item in account.get("balances", []) if item.get("asset") == asset),
            None,
        )
        return Decimal(str(balance.get("free", "0"))) if balance else Decimal(0)

    def query_order(self, symbol: str, client_order_id: str) -> OrderResult:
        payload = self._request(
            "GET",
            "/api/v3/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
            signed=True,
        )
        return OrderResult.from_binance(payload)

    def _reconcile_order(self, symbol: str, client_order_id: str) -> OrderResult:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.query_order(symbol, client_order_id)
            except BinanceAPIError as exc:
                last_error = exc
                if exc.code != -2013:
                    break
            except BinanceError as exc:
                last_error = exc
            time.sleep(0.5 * (attempt + 1))
        raise OrderUncertainError(client_order_id) from last_error

    def place_market_order(
        self,
        symbol: str,
        side: Signal,
        client_order_id: str,
        *,
        quantity: Decimal | None = None,
        quote_order_quantity: Decimal | None = None,
    ) -> OrderResult:
        if side not in (Signal.BUY, Signal.SELL):
            raise ValueError("market orders require BUY or SELL")
        if (quantity is None) == (quote_order_quantity is None):
            raise ValueError("provide exactly one of quantity or quote_order_quantity")
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "type": "MARKET",
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL",
        }
        if quantity is not None:
            params["quantity"] = decimal_to_string(quantity)
        if quote_order_quantity is not None:
            params["quoteOrderQty"] = decimal_to_string(quote_order_quantity)
        try:
            payload = self._request(
                "POST", "/api/v3/order", params, signed=True, allow_retries=False
            )
        except BinanceTransportError:
            return self._reconcile_order(symbol, client_order_id)
        except BinanceAPIError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                return self._reconcile_order(symbol, client_order_id)
            raise
        return OrderResult.from_binance(payload)
