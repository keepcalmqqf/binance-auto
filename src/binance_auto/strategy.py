from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from .models import Signal


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, Decimal(0)) / Decimal(len(values))


def signal_at(
    closes: Sequence[Decimal], index: int, fast_window: int, slow_window: int
) -> Signal:
    if index < slow_window:
        return Signal.HOLD
    previous_fast = _mean(closes[index - fast_window : index])
    previous_slow = _mean(closes[index - slow_window : index])
    current_fast = _mean(closes[index - fast_window + 1 : index + 1])
    current_slow = _mean(closes[index - slow_window + 1 : index + 1])
    if previous_fast <= previous_slow and current_fast > current_slow:
        return Signal.BUY
    if previous_fast >= previous_slow and current_fast < current_slow:
        return Signal.SELL
    return Signal.HOLD


def latest_signal(
    closes: Sequence[Decimal], fast_window: int, slow_window: int
) -> Signal:
    if not closes:
        return Signal.HOLD
    return signal_at(closes, len(closes) - 1, fast_window, slow_window)
