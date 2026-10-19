from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable

from .binance import BinanceAPIError, BinanceClient, OrderUncertainError
from .config import AppConfig
from .models import Candle, Fill, Mode, OrderResult, Signal
from .risk import RiskManager, RiskRejected
from .rules import RuleViolation, SymbolRules, decimal_to_string
from .storage import Storage
from .strategy import latest_signal


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaperSettlement:
    price: Decimal
    quantity: Decimal
    notional: Decimal
    fee: Decimal
    cash_after: Decimal
    position_after: Decimal
    cost_after: Decimal
    realized_pnl: Decimal | None


@dataclass(frozen=True)
class PreparedOrder:
    event_key: str
    client_order_id: str
    side: Signal
    candle: Candle
    order_quantity: Decimal | None
    quote_order_quantity: Decimal | None
    prior_position: Decimal
    prior_cost: Decimal
    paper: PaperSettlement | None


class TradingEngine:
    def __init__(
        self,
        config: AppConfig,
        mode: Mode,
        client: BinanceClient,
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

    @property
    def symbol(self) -> str:
        return self.config.binance.symbol

    @property
    def _last_processed_key(self) -> str:
        return f"last_processed:{self.mode.value}:{self.symbol}:{self.config.binance.interval}"

    @property
    def _last_order_key(self) -> str:
        return f"last_order:{self.mode.value}:{self.symbol}:{self.config.binance.interval}"

    def _client_order_id(self, event_key: str, candle: Candle, side: Signal) -> str:
        value = f"ba-{self.mode.value[0]}-{self.symbol}-{candle.close_time}-{side.value[0]}"
        if len(value) <= 36:
            return value
        digest = hashlib.sha256(event_key.encode("utf-8")).hexdigest()[:12]
        return f"ba-{self.mode.value[0]}-{candle.close_time}-{side.value[0]}-{digest}"

    @staticmethod
    def _actionable_position(quantity: Decimal, price: Decimal, rules: SymbolRules) -> bool:
        if quantity <= 0:
            return False
        if rules.min_quantity > 0 and quantity < rules.min_quantity:
            return False
        if rules.min_notional > 0 and quantity * price < rules.min_notional:
            return False
        return True

    def _prepare_order(
        self, side: Signal, candle: Candle, rules: SymbolRules, event_key: str
    ) -> PreparedOrder:
        position, cost_basis = self.storage.get_position(self.mode, self.symbol)
        free_base = position
        if self.mode == Mode.PAPER:
            available_quote = self.storage.get_paper_quote(
                self.symbol, self.config.backtest.initial_quote
            )
        else:
            account = self.client.get_account()
            available_quote = self.client.free_balance(account, rules.quote_asset)
            free_base = self.client.free_balance(account, rules.base_asset)

        reference_price = candle.close
        current_equity = available_quote + position * reference_price
        now_ms = self.client.now_ms()
        day = datetime.fromtimestamp(now_ms / 1000, timezone.utc).date().isoformat()
        day_start = self.storage.daily_start_equity(
            self.mode, self.symbol, day, current_equity
        )
        last_order_text = self.storage.get_state(self._last_order_key)
        last_order_at = int(last_order_text) if last_order_text else None
        client_order_id = self._client_order_id(event_key, candle, side)

        if side == Signal.BUY:
            if self._actionable_position(position, reference_price, rules):
                raise RiskRejected("a strategy position is already open")
            budget = self.risk.buy_budget(
                available_quote=available_quote,
                position_quantity=position,
                reference_price=reference_price,
                current_equity=current_equity,
                day_start_equity=day_start,
                last_order_at=last_order_at,
                now_ms=now_ms,
            )
            if self.mode == Mode.PAPER:
                price = reference_price * (
                    1 + self.config.backtest.slippage_bps / Decimal(10000)
                )
                spend_cap = min(
                    budget, available_quote / (1 + self.config.backtest.fee_rate)
                )
                quantity = rules.floor_quantity(spend_cap / price)
                notional = quantity * price
                fee = notional * self.config.backtest.fee_rate
                rules.validate_market_buy(notional, price)
                if quantity <= 0 or notional + fee > available_quote:
                    raise RuleViolation("paper buy exceeds cash after fees and rounding")
                settlement = PaperSettlement(
                    price=price,
                    quantity=quantity,
                    notional=notional,
                    fee=fee,
                    cash_after=available_quote - notional - fee,
                    position_after=position + quantity,
                    cost_after=cost_basis + notional + fee,
                    realized_pnl=None,
                )
                return PreparedOrder(
                    event_key,
                    client_order_id,
                    side,
                    candle,
                    None,
                    None,
                    position,
                    cost_basis,
                    settlement,
                )

            quote_quantity = rules.floor_quote(budget)
            rules.validate_market_buy(quote_quantity, reference_price)
            return PreparedOrder(
                event_key,
                client_order_id,
                side,
                candle,
                None,
                quote_quantity,
                position,
                cost_basis,
                None,
            )

        if not self._actionable_position(position, reference_price, rules):
            raise RiskRejected("there is no actionable strategy position to sell")
        quantity = rules.floor_quantity(position)
        if self.mode != Mode.PAPER:
            quantity = min(quantity, rules.floor_quantity(free_base))
        price = reference_price
        if self.mode == Mode.PAPER:
            price = reference_price * (
                1 - self.config.backtest.slippage_bps / Decimal(10000)
            )
        rules.validate_market_sell(quantity, price)

        paper: PaperSettlement | None = None
        if self.mode == Mode.PAPER:
            notional = quantity * price
            fee = notional * self.config.backtest.fee_rate
            cost_reduction = cost_basis * (quantity / position)
            position_after = position - quantity
            cost_after = cost_basis - cost_reduction
            if position_after == 0:
                cost_after = Decimal(0)
            paper = PaperSettlement(
                price=price,
                quantity=quantity,
                notional=notional,
                fee=fee,
                cash_after=available_quote + notional - fee,
                position_after=position_after,
                cost_after=cost_after,
                realized_pnl=notional - fee - cost_reduction,
            )
        return PreparedOrder(
            event_key,
            client_order_id,
            side,
            candle,
            quantity if self.mode != Mode.PAPER else None,
            None,
            position,
            cost_basis,
            paper,
        )

    def _execute_paper(self, prepared: PreparedOrder, rules: SymbolRules) -> OrderResult:
        settlement = prepared.paper
        if settlement is None:
            raise EngineError("paper settlement is missing")
        order = OrderResult(
            client_order_id=prepared.client_order_id,
            order_id=f"paper-{prepared.candle.close_time}-{prepared.side.value}",
            symbol=self.symbol,
            side=prepared.side,
            status="FILLED",
            executed_quantity=settlement.quantity,
            quote_quantity=settlement.notional,
            transact_time=prepared.candle.close_time,
            fills=(
                Fill(
                    price=settlement.price,
                    quantity=settlement.quantity,
                    commission=settlement.fee,
                    commission_asset=rules.quote_asset,
                    trade_id=prepared.client_order_id,
                ),
            ),
        )
        self.storage.record_order(order, self.mode)
        self.storage.record_fill(order.client_order_id, order.fills[0])
        self.storage.set_paper_quote(self.symbol, settlement.cash_after)
        self.storage.save_position(
            self.mode, self.symbol, settlement.position_after, settlement.cost_after
        )
        return order

    def _update_exchange_position(self, prepared: PreparedOrder, order: OrderResult) -> None:
        executed = order.executed_quantity
        if executed <= 0:
            return
        if prepared.side == Signal.BUY:
            base_commission = sum(
                (
                    fill.commission
                    for fill in order.fills
                    if fill.commission_asset == self._rules.base_asset
                ),
                Decimal(0),
            )
            quote_commission = sum(
                (
                    fill.commission
                    for fill in order.fills
                    if fill.commission_asset == self._rules.quote_asset
                ),
                Decimal(0),
            )
            new_position = prepared.prior_position + max(Decimal(0), executed - base_commission)
            new_cost = prepared.prior_cost + order.quote_quantity + quote_commission
        else:
            ratio = min(Decimal(1), executed / prepared.prior_position)
            new_position = max(Decimal(0), prepared.prior_position - executed)
            new_cost = prepared.prior_cost * (1 - ratio)
            if new_position == 0:
                new_cost = Decimal(0)
        self.storage.save_position(self.mode, self.symbol, new_position, new_cost)

    def _execute_exchange(self, prepared: PreparedOrder) -> OrderResult:
        order = self.client.place_market_order(
            self.symbol,
            prepared.side,
            prepared.client_order_id,
            quantity=prepared.order_quantity,
            quote_order_quantity=prepared.quote_order_quantity,
        )
        self.storage.record_order(order, self.mode)
        for fill in order.fills:
            self.storage.record_fill(order.client_order_id, fill)
        if order.status not in {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}:
            raise EngineError(
                f"order {order.client_order_id} is still {order.status}; reconcile before restart"
            )
        self._update_exchange_position(prepared, order)
        return order

    def _claim(self, prepared: PreparedOrder) -> bool:
        return self.storage.claim_signal(
            event_key=prepared.event_key,
            client_order_id=prepared.client_order_id,
            mode=self.mode,
            symbol=self.symbol,
            candle_close_time=prepared.candle.close_time,
            side=prepared.side,
            now_ms=self.client.now_ms(),
        )

    def _skipped_order(
        self,
        *,
        event_key: str,
        client_order_id: str,
        side: Signal,
        candle: Candle,
        reason: str,
    ) -> None:
        prepared = PreparedOrder(
            event_key,
            client_order_id,
            side,
            candle,
            None,
            None,
            Decimal(0),
            Decimal(0),
            None,
        )
        if not self._claim(prepared):
            raise EngineError(f"signal was claimed concurrently: {event_key}")
        self.storage.finish_signal(event_key, "SKIPPED", reason)
        self.storage.set_state(self._last_processed_key, str(candle.close_time))
        self.output(f"{candle.close_time} {side.value} skipped: {reason}")

    def poll_once(self, rules: SymbolRules) -> None:
        self._rules = rules
        candles = self.client.get_recent_closed_klines(
            self.symbol,
            self.config.binance.interval,
            limit=self.config.strategy.slow_window + 2,
        )
        if len(candles) < self.config.strategy.slow_window + 1:
            raise EngineError("not enough closed candles to initialize the strategy")
        latest = candles[-1]
        last_text = self.storage.get_state(self._last_processed_key)
        if last_text is None:
            self.storage.set_state(self._last_processed_key, str(latest.close_time))
            self.output(
                f"initialized at closed candle {latest.close_time}; "
                "waiting for the next candle before trading"
            )
            return
        last_processed = int(last_text)
        if latest.close_time <= last_processed:
            self.output(f"no new closed candle after {last_processed}")
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
            self.output(f"{latest.close_time} HOLD close={decimal_to_string(latest.close)}")
            return

        event_key = (
            f"{self.mode.value}:{self.symbol}:{self.config.binance.interval}:"
            f"{latest.close_time}:{signal.value}"
        )
        existing = self.storage.get_signal(event_key)
        if existing is not None:
            if existing["status"] == "PENDING":
                raise EngineError(
                    f"unresolved signal {event_key} with clientOrderId="
                    f"{existing['client_order_id']}; refusing to risk a duplicate order"
                )
            self.storage.set_state(self._last_processed_key, str(latest.close_time))
            return

        client_order_id = self._client_order_id(event_key, latest, signal)
        try:
            prepared = self._prepare_order(signal, latest, rules, event_key)
        except (RiskRejected, RuleViolation) as exc:
            self._skipped_order(
                event_key=event_key,
                client_order_id=client_order_id,
                side=signal,
                candle=latest,
                reason=str(exc),
            )
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
            f"quote={decimal_to_string(order.quote_quantity)}"
        )

    def run(self, *, once: bool = False) -> None:
        rules = self.client.get_symbol_rules(self.symbol)
        self.client.synchronize_time()
        while True:
            self.poll_once(rules)
            if once:
                return
            time.sleep(self.config.binance.poll_seconds)
