from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import typer
from dotenv import load_dotenv

from .backtest import BacktestError, BacktestResult, Backtester
from .binance import BinanceClient, BinanceError
from .config import AppConfig, ConfigError, load_config
from .engine import EngineError, TradingEngine
from .models import Mode
from .rules import RuleViolation, decimal_to_string
from .storage import Storage


app = typer.Typer(
    name="binance-auto",
    help="Risk-controlled Binance Spot backtesting, paper trading, and automation.",
    no_args_is_help=True,
    add_completion=False,
)
DEFAULT_CONFIG = Path("config.toml")


@app.callback()
def callback(
    ctx: typer.Context,
    config: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="TOML configuration file"
    ),
) -> None:
    ctx.obj = config


def _settings(ctx: typer.Context) -> AppConfig:
    path = Path(ctx.obj or DEFAULT_CONFIG)
    return load_config(path, allow_missing=path == DEFAULT_CONFIG)


def _credentials(mode: Mode) -> tuple[str | None, str | None]:
    prefix = "BINANCE_TESTNET" if mode == Mode.TESTNET else "BINANCE"
    return os.getenv(f"{prefix}_API_KEY"), os.getenv(f"{prefix}_API_SECRET")


def _client(config: AppConfig, mode: Mode, *, require_credentials: bool) -> BinanceClient:
    api_key: str | None = None
    api_secret: str | None = None
    if mode != Mode.PAPER:
        api_key, api_secret = _credentials(mode)
        if require_credentials and (not api_key or not api_secret):
            prefix = "BINANCE_TESTNET" if mode == Mode.TESTNET else "BINANCE"
            raise ConfigError(
                f"missing {prefix}_API_KEY or {prefix}_API_SECRET environment variable"
            )
    base_url = (
        config.binance.testnet_url if mode == Mode.TESTNET else config.binance.production_url
    )
    return BinanceClient(
        base_url,
        api_key=api_key,
        api_secret=api_secret,
        timeout_seconds=config.binance.timeout_seconds,
        recv_window=config.binance.recv_window,
    )


def _percent(value: Decimal) -> str:
    return f"{value * 100:.2f}%"


def _print_backtest(result: BacktestResult, quote_asset: str, base_asset: str) -> None:
    rows = [
        ("Initial quote", f"{result.initial_quote:.2f} {quote_asset}"),
        ("Final equity", f"{result.final_equity:.2f} {quote_asset}"),
        ("Strategy return", _percent(result.total_return)),
        ("Buy & hold return", _percent(result.buy_hold_return)),
        ("Max drawdown", _percent(result.max_drawdown)),
        ("Orders", str(len(result.trades))),
        ("Closed round trips", str(result.closed_trades)),
        ("Win rate", _percent(result.win_rate)),
        ("Fees", f"{decimal_to_string(result.total_fees)} {quote_asset}"),
        ("Skipped orders", str(result.skipped_orders)),
        ("Ending base quantity", f"{decimal_to_string(result.ending_base_quantity)} {base_asset}"),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        typer.echo(f"{label:<{width}}  {value}")


def _export_trades(result: BacktestResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "signal_close_time",
                "execution_time",
                "side",
                "price",
                "quantity",
                "notional",
                "fee",
                "realized_pnl",
            ]
        )
        for trade in result.trades:
            writer.writerow(
                [
                    trade.signal_close_time,
                    trade.execution_time,
                    trade.side.value,
                    decimal_to_string(trade.price),
                    decimal_to_string(trade.quantity),
                    decimal_to_string(trade.notional),
                    decimal_to_string(trade.fee),
                    "" if trade.realized_pnl is None else decimal_to_string(trade.realized_pnl),
                ]
            )


def _date_boundary(value: str, *, end: bool = False) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ConfigError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc
    return parsed + timedelta(days=1) - timedelta(milliseconds=1) if end else parsed


@app.command()
def doctor(
    ctx: typer.Context,
    mode: Mode = typer.Option(Mode.PAPER, "--mode", "-m"),
) -> None:
    """Validate configuration and connectivity without placing orders."""
    config = _settings(ctx)
    typer.echo(f"mode={mode.value} symbol={config.binance.symbol} interval={config.binance.interval}")
    with _client(config, mode, require_credentials=False) as client:
        server_time = client.synchronize_time()
        rules = client.get_symbol_rules(config.binance.symbol)
        candles = client.get_recent_closed_klines(
            config.binance.symbol, config.binance.interval, limit=2
        )
        typer.echo(f"public API: OK serverTime={server_time}")
        typer.echo(
            f"symbol rules: {rules.base_asset}/{rules.quote_asset} "
            f"step={decimal_to_string(rules.step_size)} "
            f"minNotional={decimal_to_string(rules.min_notional)}"
        )
        if candles:
            typer.echo(
                f"latest closed candle: {candles[-1].close_time} "
                f"close={decimal_to_string(candles[-1].close)}"
            )
        if mode != Mode.PAPER:
            api_key, api_secret = _credentials(mode)
            if api_key and api_secret:
                client.get_account()
                typer.echo("signed account read: OK (no order placed)")
            else:
                typer.echo("signed account read: SKIPPED (credentials not set)")
    typer.echo("configuration and connectivity checks passed")


@app.command()
def backtest(
    ctx: typer.Context,
    start: str | None = typer.Option(None, "--start", help="UTC date: YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="UTC date: YYYY-MM-DD"),
    days: int = typer.Option(180, "--days", min=1),
    export: Path | None = typer.Option(None, "--export", help="Write fills to CSV"),
) -> None:
    """Backtest the configured strategy on real Binance public klines."""
    config = _settings(ctx)
    with _client(config, Mode.PAPER, require_credentials=False) as client:
        client.synchronize_time()
        now = datetime.fromtimestamp(client.now_ms() / 1000, timezone.utc)
        end_dt = _date_boundary(end, end=True) if end else now
        end_dt = min(end_dt, now)
        start_dt = (
            _date_boundary(start)
            if start
            else end_dt - timedelta(days=days)
        )
        if start_dt >= end_dt:
            raise ConfigError("backtest start must be earlier than end")
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        rules = client.get_symbol_rules(config.binance.symbol)
        candles = client.get_historical_klines(
            config.binance.symbol,
            config.binance.interval,
            start_time=start_ms,
            end_time=end_ms,
        )
        candles = [candle for candle in candles if candle.close_time < client.now_ms()]
    typer.echo(f"fetched {len(candles)} closed candles")
    result = Backtester(config.strategy, config.backtest, config.risk, rules).run(candles)
    _print_backtest(result, rules.quote_asset, rules.base_asset)
    if export:
        _export_trades(result, export)
        typer.echo(f"exported {len(result.trades)} fills to {export}")


@app.command("run")
def run_command(
    ctx: typer.Context,
    mode: Mode = typer.Option(Mode.PAPER, "--mode", "-m"),
    once: bool = typer.Option(False, "--once", help="Poll once and exit"),
    confirm_live: str | None = typer.Option(
        None, "--confirm-live", help="Required symbol confirmation for real orders"
    ),
) -> None:
    """Run paper, Spot Testnet, or explicitly unlocked live trading."""
    config = _settings(ctx)
    with _client(config, mode, require_credentials=True) as client:
        with Storage(config.storage.path) as storage:
            engine = TradingEngine(
                config,
                mode,
                client,
                storage,
                output=typer.echo,
                confirm_live=confirm_live,
            )
            engine.run(once=once)


def main() -> None:
    load_dotenv()
    try:
        app()
    except (ConfigError, BinanceError, BacktestError, EngineError, RuleViolation) as exc:
        typer.echo(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        typer.echo("stopped", file=sys.stderr)
        raise SystemExit(130) from None
