from __future__ import annotations

from decimal import Decimal

from .config import RiskConfig


class RiskRejected(RuntimeError):
    pass


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config

    def buy_budget(
        self,
        *,
        available_quote: Decimal,
        position_quantity: Decimal,
        reference_price: Decimal,
        current_equity: Decimal,
        day_start_equity: Decimal,
        last_order_at: int | None,
        now_ms: int,
    ) -> Decimal:
        daily_loss = day_start_equity - current_equity
        if daily_loss >= self.config.max_daily_loss_quote:
            raise RiskRejected(
                f"daily loss limit reached: {daily_loss} >= "
                f"{self.config.max_daily_loss_quote}"
            )
        if last_order_at is not None and self.config.cooldown_minutes > 0:
            cooldown_ms = self.config.cooldown_minutes * 60_000
            if now_ms - last_order_at < cooldown_ms:
                raise RiskRejected("order cooldown is still active")
        position_room = (
            current_equity * self.config.max_position_pct
            - position_quantity * reference_price
        )
        budget = min(self.config.max_order_quote, available_quote, position_room)
        if budget <= 0:
            raise RiskRejected("no available quote balance or position capacity")
        return budget
