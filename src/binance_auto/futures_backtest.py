from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from .backtest import BacktestError, BacktestResult, Trade
from .config import BacktestConfig, FuturesConfig, RiskConfig, StrategyConfig
from .futures_accounting import settle_target_position, unrealized_pnl
from .models import Candle, Signal
from .risk import RiskManager, RiskRejected
from .rules import RuleViolation, SymbolRules
from .strategy import signal_at


class FuturesBacktester:
    def __init__(
        self,
        strategy: StrategyConfig,
        settings: BacktestConfig,
        risk: RiskConfig,
        futures: FuturesConfig,
        rules: SymbolRules,
    ):
        self.strategy = strategy
        self.settings = settings
        self.risk = RiskManager(risk)
        self.futures = futures
        self.rules = rules

    def _execution_price(self, side: Signal, price: Decimal) -> Decimal:
        slippage = self.settings.slippage_bps / Decimal(10000)
        return price * (1 + slippage) if side == Signal.BUY else price * (1 - slippage)

    def run(self, candles: list[Candle]) -> BacktestResult:
        if len(candles) < self.strategy.slow_window + 2:
            raise BacktestError(
                f"need at least {self.strategy.slow_window + 2} closed candles; "
                f"received {len(candles)}"
            )
        if any(candles[i].open_time >= candles[i + 1].open_time for i in range(len(candles) - 1)):
            raise BacktestError("candles must be strictly sorted by open time")

        initial = self.settings.initial_quote
        wallet = initial
        position = Decimal(0)
        entry_price = Decimal(0)
        trades: list[Trade] = []
        skipped = 0
        closes = [candle.close for candle in candles]
        equity_curve = [initial]
        last_order_at: int | None = None
        current_day = ""
        day_start_equity = initial

        for index in range(len(candles) - 1):
            signal = signal_at(
                closes, index, self.strategy.fast_window, self.strategy.slow_window
            )
            signal_candle = candles[index]
            execution_candle = candles[index + 1]
            if signal != Signal.HOLD:
                direction = Decimal(1) if signal == Signal.BUY else Decimal(-1)
                if position * direction <= 0:
                    mark_price = execution_candle.open
                    current_equity = wallet + unrealized_pnl(position, entry_price, mark_price)
                    day = datetime.fromtimestamp(
                        execution_candle.open_time / 1000, timezone.utc
                    ).date().isoformat()
                    if day != current_day:
                        current_day = day
                        day_start_equity = current_equity
                    price = self._execution_price(signal, mark_price)
                    target_position = Decimal(0)
                    try:
                        target_notional = self.risk.futures_target_notional(
                            available_margin=current_equity,
                            current_equity=current_equity,
                            leverage=self.futures.leverage,
                            day_start_equity=day_start_equity,
                            last_order_at=last_order_at,
                            now_ms=execution_candle.open_time,
                        )
                        target_quantity = self.rules.floor_quantity(target_notional / price)
                        target_position = direction * target_quantity
                    except RiskRejected:
                        if position == 0:
                            skipped += 1
                            equity_curve.append(wallet)
                            continue

                    if target_position != position:
                        try:
                            settlement = settle_target_position(
                                position, entry_price, target_position, price
                            )
                            self.rules.validate_quantity(settlement.order_quantity, price)
                        except (RuleViolation, ValueError):
                            if position == 0 or target_position == 0:
                                skipped += 1
                                equity_curve.append(
                                    wallet
                                    + unrealized_pnl(position, entry_price, execution_candle.close)
                                )
                                continue
                            target_position = Decimal(0)
                            settlement = settle_target_position(
                                position, entry_price, target_position, price
                            )
                            try:
                                self.rules.validate_quantity(settlement.order_quantity, price)
                            except RuleViolation:
                                skipped += 1
                                equity_curve.append(
                                    wallet
                                    + unrealized_pnl(position, entry_price, execution_candle.close)
                                )
                                continue

                        notional = settlement.order_quantity * price
                        fee = notional * self.settings.fee_rate
                        wallet += settlement.realized_pnl - fee
                        position = settlement.position_after
                        entry_price = settlement.entry_price_after
                        last_order_at = execution_candle.open_time
                        trades.append(
                            Trade(
                                signal_candle.close_time,
                                execution_candle.open_time,
                                settlement.side,
                                price,
                                settlement.order_quantity,
                                notional,
                                fee,
                                settlement.realized_pnl
                                if settlement.closed_quantity > 0
                                else None,
                            )
                        )

            equity_curve.append(wallet + unrealized_pnl(position, entry_price, execution_candle.close))

        final_equity = wallet + unrealized_pnl(position, entry_price, candles[-1].close)
        peak = equity_curve[0]
        max_drawdown = Decimal(0)
        for equity in equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)

        benchmark_price = self._execution_price(Signal.BUY, candles[0].open)
        benchmark_quantity = self.rules.floor_quantity(
            (initial / (1 + self.settings.fee_rate)) / benchmark_price
        )
        benchmark_notional = benchmark_quantity * benchmark_price
        benchmark_fee = benchmark_notional * self.settings.fee_rate
        benchmark_cash = initial - benchmark_notional - benchmark_fee
        benchmark_equity = benchmark_cash + benchmark_quantity * candles[-1].close

        closed_trades = sum(trade.realized_pnl is not None for trade in trades)
        winning = sum(
            trade.realized_pnl is not None and trade.realized_pnl > 0 for trade in trades
        )
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
            ending_base_quantity=position,
            trades=tuple(trades),
        )
