from __future__ import annotations

import sqlite3
import time
from decimal import Decimal
from pathlib import Path

from .models import Fill, Mode, OrderResult, Signal
from .rules import decimal_to_string


def _scope_value(scope: Mode | str) -> str:
    return scope.value if isinstance(scope, Mode) else scope


def _paper_account_key(symbol: str, scope: Mode | str | None) -> str:
    return symbol if scope is None else f"{_scope_value(scope)}:{symbol}"


class Storage:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _create_tables(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_accounts (
                    symbol TEXT PRIMARY KEY,
                    quote_balance TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    base_quantity TEXT NOT NULL,
                    cost_basis TEXT NOT NULL,
                    PRIMARY KEY (mode, symbol)
                );
                CREATE TABLE IF NOT EXISTS daily_equity (
                    mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    day TEXT NOT NULL,
                    start_equity TEXT NOT NULL,
                    PRIMARY KEY (mode, symbol, day)
                );
                CREATE TABLE IF NOT EXISTS signals (
                    event_key TEXT PRIMARY KEY,
                    client_order_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    candle_close_time INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    client_order_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    executed_quantity TEXT NOT NULL,
                    quote_quantity TEXT NOT NULL,
                    transact_time INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_order_id TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    price TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    commission TEXT NOT NULL,
                    commission_asset TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE (client_order_id, trade_id)
                );
                """
            )

    def get_state(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row else None

    def set_state(self, key: str, value: str) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_paper_quote(
        self, symbol: str, initial_quote: Decimal, *, scope: Mode | str | None = None
    ) -> Decimal:
        key = _paper_account_key(symbol, scope)
        row = self._connection.execute(
            "SELECT quote_balance FROM paper_accounts WHERE symbol = ?", (key,)
        ).fetchone()
        if row is None:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO paper_accounts(symbol, quote_balance) VALUES(?, ?)",
                    (key, decimal_to_string(initial_quote)),
                )
            return initial_quote
        return Decimal(str(row["quote_balance"]))

    def set_paper_quote(
        self,
        symbol: str,
        quote_balance: Decimal,
        *,
        scope: Mode | str | None = None,
    ) -> None:
        key = _paper_account_key(symbol, scope)
        with self._connection:
            self._connection.execute(
                "UPDATE paper_accounts SET quote_balance = ? WHERE symbol = ?",
                (decimal_to_string(quote_balance), key),
            )

    def get_position(self, mode: Mode | str, symbol: str) -> tuple[Decimal, Decimal]:
        row = self._connection.execute(
            "SELECT base_quantity, cost_basis FROM positions WHERE mode = ? AND symbol = ?",
            (_scope_value(mode), symbol),
        ).fetchone()
        if row is None:
            return Decimal(0), Decimal(0)
        return Decimal(str(row["base_quantity"])), Decimal(str(row["cost_basis"]))

    def save_position(
        self, mode: Mode | str, symbol: str, base_quantity: Decimal, cost_basis: Decimal
    ) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO positions(mode, symbol, base_quantity, cost_basis) "
                "VALUES(?, ?, ?, ?) ON CONFLICT(mode, symbol) DO UPDATE SET "
                "base_quantity = excluded.base_quantity, cost_basis = excluded.cost_basis",
                (
                    _scope_value(mode),
                    symbol,
                    decimal_to_string(base_quantity),
                    decimal_to_string(cost_basis),
                ),
            )

    def daily_start_equity(
        self, mode: Mode | str, symbol: str, day: str, current_equity: Decimal
    ) -> Decimal:
        row = self._connection.execute(
            "SELECT start_equity FROM daily_equity WHERE mode = ? AND symbol = ? AND day = ?",
            (_scope_value(mode), symbol, day),
        ).fetchone()
        if row is None:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO daily_equity(mode, symbol, day, start_equity) "
                    "VALUES(?, ?, ?, ?)",
                    (_scope_value(mode), symbol, day, decimal_to_string(current_equity)),
                )
            return current_equity
        return Decimal(str(row["start_equity"]))

    def get_signal(self, event_key: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM signals WHERE event_key = ?", (event_key,)
        ).fetchone()

    def claim_signal(
        self,
        *,
        event_key: str,
        client_order_id: str,
        mode: Mode | str,
        symbol: str,
        candle_close_time: int,
        side: Signal,
        now_ms: int,
    ) -> bool:
        with self._connection:
            cursor = self._connection.execute(
                "INSERT OR IGNORE INTO signals(" 
                "event_key, client_order_id, mode, symbol, candle_close_time, side, "
                "status, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)",
                (
                    event_key,
                    client_order_id,
                    _scope_value(mode),
                    symbol,
                    candle_close_time,
                    side.value,
                    now_ms,
                    now_ms,
                ),
            )
        return cursor.rowcount == 1

    def finish_signal(self, event_key: str, status: str, detail: str = "") -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE signals SET status = ?, detail = ?, updated_at = ? WHERE event_key = ?",
                (status, detail, int(time.time() * 1000), event_key),
            )

    def record_order(self, order: OrderResult, mode: Mode | str) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO orders(" 
                "client_order_id, order_id, mode, symbol, side, status, "
                "executed_quantity, quote_quantity, transact_time, updated_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(client_order_id) DO UPDATE SET "
                "order_id = excluded.order_id, status = excluded.status, "
                "executed_quantity = excluded.executed_quantity, "
                "quote_quantity = excluded.quote_quantity, "
                "transact_time = excluded.transact_time, updated_at = excluded.updated_at",
                (
                    order.client_order_id,
                    order.order_id,
                    _scope_value(mode),
                    order.symbol,
                    order.side.value,
                    order.status,
                    decimal_to_string(order.executed_quantity),
                    decimal_to_string(order.quote_quantity),
                    order.transact_time,
                    int(time.time() * 1000),
                ),
            )

    def record_fill(self, client_order_id: str, fill: Fill) -> None:
        trade_id = fill.trade_id or f"{decimal_to_string(fill.price)}:{decimal_to_string(fill.quantity)}"
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO fills(" 
                "client_order_id, trade_id, price, quantity, commission, "
                "commission_asset, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    client_order_id,
                    trade_id,
                    decimal_to_string(fill.price),
                    decimal_to_string(fill.quantity),
                    decimal_to_string(fill.commission),
                    fill.commission_asset,
                    int(time.time() * 1000),
                ),
            )
