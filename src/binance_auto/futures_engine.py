from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable

from .binance import BinanceAPIError, OrderUncertainError
from .config import AppConfig
from .engine import EngineError
from .futures import BinanceFuturesClient
from .futures_accounting import settle_target_position, unrealized_pnl
from .models import Candle, Fill, Mode, OrderResult, Signal
from .risk import RiskManager, RiskRejected
from .rules import RuleViolation, SymbolRules, decimal_to_string
from .storage import Storage
from .strategy import latest_signal


@dataclass(frozen=True)
class EngineAccount:
    wallet_balance: Decimal
    current_equity: Decimal
    available_margin: Decimal
    position: Decimal
    entry_price: Decimal


@dataclass(frozen=True)
class PreparedFuturesOrder:
    event_key: str
    client_order_id: str
    side: Signal
    candle: Candle
    quantity: Decimal
    reduce_only: bool
    target_position: Decimal
    execution_price: Decimal
    wallet_before: Decimal
    position_before: Decimal
    entry_before: Decimal


class FuturesTradingEngine:
    def __init__(
        self,
        config: AppConfig,
        mode: Mode,
        client: BinanceFuturesClient,
        storage: Storage,
        *,
        output: Callable[[str], None] = print,
        confirm_live: str | None = None,
    ):
        if mode == Mode.LIVE:
            if not config.live.enabled:
                raise EngineError("live trading is disabled: set live.enabled=true explicitly")
            if os.getenv("BINANCE_AUTO_ENABLE_LIVE") != "YES":
                raise EngineError("live trading requires BINANCE_AUTO_ENABLE_LIVE=YES")
            if confirm_live != config.binance.symbol:
                raise EngineError(
                    f"live trading requires --confirm-live {config.binance.symbol}"
                )
        self.config = config
        self.mode = mode
        self.client = client
        self.storage = storage
        self.output = output
        self.risk = RiskManager(config.risk)
        self.scope = f"futures:{mode.value}"

    @property
    def symbol(self) -> str:
        return self.config.binance.symbol

    @property
    def _last_processed_key(self) -> str:
        return f"last_processed:{self.scope}:{self.symbol}:{self.config.binance.interval}"

    @property
    def _last_order_key(self) -> str:
        return f"last_order:{self.scope}:{self.symbol}:{self.config.binance.interval}"

    def _client_order_id(self, event_key: str, candle: Candle, side: Signal) -> str:
        value = f"bf-{self.mode.value[0]}-{self.symbol}-{candle.close_time}-{side.value[0]}"
        if len(value) <= 36:
            return value
        digest = hashlib.sha256(event_key.encode("utf-8")).hexdigest()[:12]
        return f"bf-{self.mode.value[0]}-{candle.close_time}-{side.value[0]}-{digest}"

    def _assert_position_ownership(self) -> None:
        if self.mode == Mode.PAPER:
            return
        actual = self.client.get_account_state(self.symbol)
        stored, _ = self.storage.get_position(self.scope, self.symbol)
        if actual.position_quantity != stored:
            raise EngineError(
                f"actual futures position {actual.position_quantity} differs from strategy "
                f"state {stored} for {self.symbol}; refusing to manage an external position"
            )

    def _account(self, reference_price: Decimal) -> EngineAccount:
        if self.mode == Mode.PAPER:
            wallet = self.storage.get_paper_quote(
                self.symbol, self.config.backtest.initial_quote, scope=self.scope
            )
            position, cost_basis = self.storage.get_position(self.scope, self.symbol)
            entry = cost_basis / abs(position) if position else Decimal(0)
            equity = wallet + unrealized_pnl(position, entry, reference_price)
            return EngineAccount(wallet, equity, max(Decimal(0), equity), position, entry)

        actual = self.client.get_account_state(self.symbol)
        stored, _ = self.storage.get_position(self.scope, self.symbol)
        if actual.position_quantity != stored:
            raise EngineError(
                f"actual futures position {actual.position_quantity} differs from strategy "
                f"state {stored} for {self.symbol}; refusing to trade"
            )
        released_margin = (
            abs(actual.position_quantity)
            * actual.entry_price
            / Decimal(self.config.futures.leverage)
        )
        available = max(Decimal(0), actual.available_balance + released_margin)
        return EngineAccount(
            actual.wallet_balance,
            actual.margin_balance,
            available,
            actual.position_quantity,
            actual.entry_price,
        )

    def _prepare_order(
        self, signal: Signal, candle: Candle, rules: SymbolRules, event_key: str
    ) -> PreparedFuturesOrder:
        account = self._account(candle.close)
        direction = Decimal(1) if signal == Signal.BUY else Decimal(-1)
        if account.position * direction > 0:
            raise RiskRejected("the futures position already matches the signal direction")

        now_ms = self.client.now_ms()
        day = datetime.fromtimestamp(now_ms / 1000, timezone.utc).date().isoformat()
        day_start = self.storage.daily_start_equity(
            self.scope, self.symbol, day, account.current_equity
        )
        last_order_text = self.storage.get_state(self._last_order_key)
        last_order_at = int(last_order_text) if last_order_text else None
        target = Decimal(0)
        try:
            target_notional = self.risk.futures_target_notional(
                available_margin=account.available_margin,
                current_equity=account.current_equity,
                leverage=self.config.futures.leverage,
                day_start_equity=day_start,
                last_order_at=last_order_at,
                now_ms=now_ms,
            )
            target = direction * rules.floor_quantity(target_notional / candle.close)
            if target:
                rules.validate_quantity(abs(target), candle.close)
        except RiskRejected:
            if account.position == 0:
                raise
        except RuleViolation:
            if account.position == 0:
                raise
            target = Decimal(0)

        if target == account.position:
            raise RiskRejected("the calculated futures target does not change the position")
        delta = target - account.position
        side = Signal.BUY if delta > 0 else Signal.SELL
        slippage = self.config.backtest.slippage_bps / Decimal(10000)
        execution_price = candle.close * (1 + slippage if side == Signal.BUY else 1 - slippage)
        quantity = rules.floor_quantity(abs(delta))
        reduce_only = target == 0 and account.position != 0
        try:
            rules.validate_quantity(quantity, execution_price)
        except RuleViolation:
            if target == 0 or account.position == 0:
                raise
            target = Decimal(0)
            delta = target - account.position
            side = Signal.BUY if delta > 0 else Signal.SELL
            execution_price = candle.close * (
                1 + slippage if side == Signal.BUY else 1 - slippage
            )
            quantity = rules.floor_quantity(abs(delta))
            reduce_only = True
            rules.validate_quantity(quantity, execution_price)

        return PreparedFuturesOrder(
            event_key=event_key,
            client_order_id=self._client_order_id(event_key, candle, side),
            side=side,
            candle=candle,
            quantity=quantity,
            reduce_only=reduce_only,
            target_position=target,
            execution_price=execution_price,
            wallet_before=account.wallet_balance,
            position_before=account.position,
            entry_before=account.entry_price,
        )

    def _execute_paper(self, prepared: PreparedFuturesOrder, rules: SymbolRules) -> OrderResult:
        settlement = settle_target_position(
            prepared.position_before,
            prepared.entry_before,
            prepared.target_position,
            prepared.execution_price,
        )
        if settlement.order_quantity != prepared.quantity:
            raise EngineError("paper futures quantity changed during settlement")
        notional = prepared.quantity * prepared.execution_price
        fee = notional * self.config.backtest.fee_rate
        wallet_after = prepared.wallet_before + settlement.realized_pnl - fee
        order = OrderResult(
            client_order_id=prepared.client_order_id,
            order_id=f"paper-futures-{prepared.candle.close_time}-{prepared.side.value}",
            symbol=self.symbol,
            side=prepared.side,
            status="FILLED",
            executed_quantity=prepared.quantity,
            quote_quantity=notional,
            transact_time=prepared.candle.close_time,
            fills=(
                Fill(
                    price=prepared.execution_price,
                    quantity=prepared.quantity,
                    commission=fee,
                    commission_asset=rules.quote_asset,
                    trade_id=prepared.client_order_id,
                ),
            ),
        )
        self.storage.record_order(order, self.scope)
        self.storage.record_fill(order.client_order_id, order.fills[0])
        self.storage.set_paper_quote(self.symbol, wallet_after, scope=self.scope)
        self.storage.save_position(
            self.scope,
            self.symbol,
            settlement.position_after,
            abs(settlement.position_after) * settlement.entry_price_after,
        )
        return order

    def _execute_exchange(self, prepared: PreparedFuturesOrder) -> OrderResult:
        order = self.client.place_market_order(
            self.symbol,
            prepared.side,
            prepared.client_order_id,
            quantity=prepared.quantity,
            reduce_only=prepared.reduce_only,
        )
        self.storage.record_order(order, self.scope)
        for fill in order.fills:
            self.storage.record_fill(order.client_order_id, fill)
        if order.status not in {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}:
            raise EngineError(
                f"futures order {order.client_order_id} is still {order.status}; reconcile before restart"
            )
        if order.executed_quantity > 0:
            actual = self.client.get_account_state(self.symbol)
            self.storage.save_position(
                self.scope,
                self.symbol,
                actual.position_quantity,
                abs(actual.position_quantity) * actual.entry_price,
            )
        return order

    def _claim(self, prepared: PreparedFuturesOrder) -> bool:
        return self.storage.claim_signal(
            event_key=prepared.event_key,
            client_order_id=prepared.client_order_id,
            mode=self.scope,
            symbol=self.symbol,
            candle_close_time=prepared.candle.close_time,
            side=prepared.side,
            now_ms=self.client.now_ms(),
        )

    def _skip(
        self,
        *,
        event_key: str,
        signal: Signal,
        candle: Candle,
        reason: str,
    ) -> None:
        client_order_id = self._client_order_id(event_key, candle, signal)
        if not self.storage.claim_signal(
            event_key=event_key,
            client_order_id=client_order_id,
            mode=self.scope,
            symbol=self.symbol,
            candle_close_time=candle.close_time,
            side=signal,
            now_ms=self.client.now_ms(),
        ):
            raise EngineError(f"signal was claimed concurrently: {event_key}")
        self.storage.finish_signal(event_key, "SKIPPED", reason)
        self.storage.set_state(self._last_processed_key, str(candle.close_time))
        self.output(f"{candle.close_time} {signal.value} skipped: {reason}")

    def poll_once(self, rules: SymbolRules) -> None:
        candles = self.client.get_recent_closed_klines(
            self.symbol,
            self.config.binance.interval,
            limit=self.config.strategy.slow_window + 2,
        )
        if len(candles) < self.config.strategy.slow_window + 1:
            raise EngineError("not enough closed futures candles to initialize the strategy")
        latest = candles[-1]
        last_text = self.storage.get_state(self._last_processed_key)
        if last_text is None:
            self.storage.set_state(self._last_processed_key, str(latest.close_time))
            self.output(
                f"initialized futures at closed candle {latest.close_time}; "
                "waiting for the next candle before trading"
            )
            return
        last_processed = int(last_text)
        if latest.close_time <= last_processed:
            self.output(f"no new closed futures candle after {last_processed}")
            return

        newer_count = sum(candle.close_time > last_processed for candle in candles)
        if newer_count > 1:
            self.output(f"skipping {newer_count - 1} stale candle(s); using the latest only")
        signal = latest_signal(
            [candle.close for candle in candles],
            self.config.strategy.fast_window,
            self.config.strategy.slow_window,
        )
        if signal == Signal.HOLD:
            self.storage.set_state(self._last_processed_key, str(latest.close_time))
            self.output(f"{latest.close_time} HOLD futures close={decimal_to_string(latest.close)}")
            return

        event_key = (
            f"{self.scope}:{self.symbol}:{self.config.binance.interval}:"
            f"{latest.close_time}:{signal.value}"
        )
        existing = self.storage.get_signal(event_key)
        if existing is not None:
            if existing["status"] == "PENDING":
                raise EngineError(
                    f"unresolved futures signal {event_key} with clientOrderId="
                    f"{existing['client_order_id']}; refusing to risk a duplicate order"
                )
            self.storage.set_state(self._last_processed_key, str(latest.close_time))
            return

        try:
            prepared = self._prepare_order(signal, latest, rules, event_key)
        except (RiskRejected, RuleViolation) as exc:
            self._skip(event_key=event_key, signal=signal, candle=latest, reason=str(exc))
            return

        if not self._claim(prepared):
            raise EngineError(f"signal was claimed concurrently: {event_key}")
        try:
            if self.mode == Mode.PAPER:
                order = self._execute_paper(prepared, rules)
            else:
                order = self._execute_exchange(prepared)
        except OrderUncertainError:
            raise
        except BinanceAPIError as exc:
            self.storage.finish_signal(event_key, "REJECTED", str(exc))
            self.storage.set_state(self._last_processed_key, str(latest.close_time))
            raise

        if order.executed_quantity <= 0:
            if order.status == "FILLED":
                raise EngineError(f"filled order {order.client_order_id} has zero executed quantity")
            self.storage.finish_signal(event_key, "REJECTED", order.status)
            self.storage.set_state(self._last_processed_key, str(latest.close_time))
            self.output(f"{order.client_order_id} finished as {order.status} with no fill")
            return

        self.storage.finish_signal(event_key, "DONE", order.status)
        self.storage.set_state(self._last_order_key, str(self.client.now_ms()))
        self.storage.set_state(self._last_processed_key, str(latest.close_time))
        self.output(
            f"{order.client_order_id} {order.side.value} {order.status} "
            f"qty={decimal_to_string(order.executed_quantity)} "
            f"quote={decimal_to_string(order.quote_quantity)} "
            f"reduceOnly={prepared.reduce_only}"
        )

    def run(self, *, once: bool = False) -> None:
        rules = self.client.get_symbol_rules(self.symbol)
        self.client.synchronize_time()
        if self.mode != Mode.PAPER:
            self._assert_position_ownership()
            self.client.configure_trading(
                self.symbol,
                self.config.futures.leverage,
                self.config.futures.margin_type,
            )
        while True:
            self.poll_once(rules)
            if once:
                return
            time.sleep(self.config.binance.poll_seconds)
