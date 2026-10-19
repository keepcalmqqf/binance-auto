from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .models import Signal


@dataclass(frozen=True)
class FuturesSettlement:
    side: Signal
    order_quantity: Decimal
    closed_quantity: Decimal
    realized_pnl: Decimal
    position_after: Decimal
    entry_price_after: Decimal


def unrealized_pnl(position: Decimal, entry_price: Decimal, mark_price: Decimal) -> Decimal:
    if position == 0:
        return Decimal(0)
    return position * (mark_price - entry_price)


def settle_target_position(
    position: Decimal,
    entry_price: Decimal,
    target_position: Decimal,
    execution_price: Decimal,
) -> FuturesSettlement:
    delta = target_position - position
    if delta == 0:
        raise ValueError("target position must change the current position")
    side = Signal.BUY if delta > 0 else Signal.SELL
    closed_quantity = Decimal(0)
    realized = Decimal(0)
    if position * delta < 0:
        closed_quantity = min(abs(position), abs(delta))
        direction = Decimal(1) if position > 0 else Decimal(-1)
        realized = closed_quantity * direction * (execution_price - entry_price)

    if target_position == 0:
        next_entry = Decimal(0)
    elif position == 0 or position * target_position < 0:
        next_entry = execution_price
    elif abs(target_position) > abs(position):
        opened = abs(target_position) - abs(position)
        next_entry = (
            abs(position) * entry_price + opened * execution_price
        ) / abs(target_position)
    else:
        next_entry = entry_price

    return FuturesSettlement(
        side=side,
        order_quantity=abs(delta),
        closed_quantity=closed_quantity,
        realized_pnl=realized,
        position_after=target_position,
        entry_price_after=next_entry,
    )
