from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .config import BacktestConfig, RiskConfig, StrategyConfig
from .models import Candle, Signal
from .rules import RuleViolation, SymbolRules
from .strategy import signal_at


class BacktestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Trade:
    signal_close_time: int
    execution_time: int
    side: Signal
    price: Decimal
    quantity: Decimal
    notional: Decimal
    fee: Decimal
    realized_pnl: Decimal | None = None


@dataclass(frozen=True)
class BacktestResult:
    initial_quote: Decimal
    final_equity: Decimal
    total_return: Decimal
    buy_hold_return: Decimal
    max_drawdown: Decimal
    total_fees: Decimal
    closed_trades: int
    winning_trades: int
    skipped_orders: int
    ending_base_quantity: Decimal
    trades: tuple[Trade, ...]

    @property
    def win_rate(self) -> Decimal:
        if not self.closed_trades:
            return Decimal(0)
        return Decimal(self.winning_trades) / Decimal(self.closed_trades)


class Backtester:
    def __init__(
        self,
        strategy: StrategyConfig,
        settings: BacktestConfig,
        risk: RiskConfig,
        rules: SymbolRules,
    ):
        self.strategy = strategy
        self.settings = settings
        self.risk = risk
        self.rules = rules

    def _buy_price(self, price: Decimal) -> Decimal:
        return price * (1 + self.settings.slippage_bps / Decimal(10000))

    def _sell_price(self, price: Decimal) -> Decimal:
        return price * (1 - self.settings.slippage_bps / Decimal(10000))

    def run(self, candles: list[Candle]) -> BacktestResult:
        if len(candles) < self.strategy.slow_window + 2:
            raise BacktestError(
                f"need at least {self.strategy.slow_window + 2} closed candles; "
                f"received {len(candles)}"
            )
        if any(candles[i].open_time >= candles[i + 1].open_time for i in range(len(candles) - 1)):
            raise BacktestError("candles must be strictly sorted by open time")

        initial = self.settings.initial_quote
        cash = initial
        base_quantity = Decimal(0)
        position_cost = Decimal(0)
        trades: list[Trade] = []
        skipped = 0
        winning = 0
        closes = [candle.close for candle in candles]
        equity_curve = [initial]

        for index in range(len(candles) - 1):
            signal = signal_at(
                closes, index, self.strategy.fast_window, self.strategy.slow_window
            )
            signal_candle = candles[index]
            execution_candle = candles[index + 1]

            if signal == Signal.BUY and base_quantity == 0:
                price = self._buy_price(execution_candle.open)
                position_cap = cash * self.risk.max_position_pct
                budget = min(
                    self.risk.max_order_quote,
                    position_cap,
                    cash / (1 + self.settings.fee_rate),
                )
                quantity = self.rules.floor_quantity(budget / price)
                notional = quantity * price
                fee = notional * self.settings.fee_rate
                try:
                    self.rules.validate_market_buy(notional, price)
                    if quantity <= 0 or notional + fee > cash:
                        raise RuleViolation("insufficient cash after quantity rounding and fees")
                except RuleViolation:
                    skipped += 1
                else:
                    cash -= notional + fee
                    base_quantity = quantity
                    position_cost = notional + fee
                    trades.append(
                        Trade(
                            signal_candle.close_time,
                            execution_candle.open_time,
                            Signal.BUY,
                            price,
                            quantity,
                            notional,
                            fee,
                        )
                    )

            elif signal == Signal.SELL and base_quantity > 0:
                price = self._sell_price(execution_candle.open)
                quantity = base_quantity
                notional = quantity * price
                fee = notional * self.settings.fee_rate
                try:
                    self.rules.validate_market_sell(quantity, price)
                except RuleViolation:
                    skipped += 1
                else:
                    proceeds = notional - fee
                    realized = proceeds - position_cost
                    cash += proceeds
                    base_quantity = Decimal(0)
                    position_cost = Decimal(0)
                    winning += int(realized > 0)
                    trades.append(
                        Trade(
                            signal_candle.close_time,
                            execution_candle.open_time,
                            Signal.SELL,
                            price,
                            quantity,
                            notional,
                            fee,
                            realized,
                        )
                    )

            equity_curve.append(cash + base_quantity * execution_candle.close)

        final_equity = cash + base_quantity * candles[-1].close
        peak = equity_curve[0]
        max_drawdown = Decimal(0)
        for equity in equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)

        benchmark_price = self._buy_price(candles[0].open)
        benchmark_quantity = self.rules.floor_quantity(
            (initial / (1 + self.settings.fee_rate)) / benchmark_price
        )
        benchmark_notional = benchmark_quantity * benchmark_price
        benchmark_fee = benchmark_notional * self.settings.fee_rate
        benchmark_cash = initial - benchmark_notional - benchmark_fee
        benchmark_equity = benchmark_cash + benchmark_quantity * candles[-1].close

        closed_trades = sum(trade.side == Signal.SELL for trade in trades)
        total_fees = sum((trade.fee for trade in trades), Decimal(0))
        return BacktestResult(
            initial_quote=initial,
            final_equity=final_equity,
            total_return=final_equity / initial - 1,
            buy_hold_return=benchmark_equity / initial - 1,
            max_drawdown=max_drawdown,
            total_fees=total_fees,
            closed_trades=closed_trades,
            winning_trades=winning,
            skipped_orders=skipped,
            ending_base_quantity=base_quantity,
            trades=tuple(trades),
        )
