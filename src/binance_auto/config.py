from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ConfigError(f"{name} must be a decimal number") from exc
    if not result.is_finite():
        raise ConfigError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class BinanceConfig:
    symbol: str = "BTCUSDT"
    market: str = "spot"
    interval: str = "1h"
    poll_seconds: int = 15
    timeout_seconds: float = 10.0
    recv_window: int = 5000
    production_url: str = "https://api.binance.com"
    testnet_url: str = "https://testnet.binance.vision"
    futures_production_url: str = "https://fapi.binance.com"
    futures_testnet_url: str = "https://testnet.binancefuture.com"


@dataclass(frozen=True)
class StrategyConfig:
    fast_window: int = 20
    slow_window: int = 50


@dataclass(frozen=True)
class BacktestConfig:
    initial_quote: Decimal = Decimal("10000")
    fee_rate: Decimal = Decimal("0.001")
    slippage_bps: Decimal = Decimal("5")


@dataclass(frozen=True)
class RiskConfig:
    max_order_quote: Decimal = Decimal("250")
    max_position_pct: Decimal = Decimal("0.25")
    max_daily_loss_quote: Decimal = Decimal("100")
    cooldown_minutes: int = 0


@dataclass(frozen=True)
class FuturesConfig:
    leverage: int = 2
    margin_type: str = "ISOLATED"


@dataclass(frozen=True)
class StorageConfig:
    path: Path = Path("data/binance_auto.db")


@dataclass(frozen=True)
class LiveConfig:
    enabled: bool = False


@dataclass(frozen=True)
class AppConfig:
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    futures: FuturesConfig = field(default_factory=FuturesConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Z0-9]{5,30}", self.binance.symbol):
            raise ConfigError("binance.symbol must contain 5-30 uppercase letters/digits")
        if self.binance.market not in {"spot", "futures"}:
            raise ConfigError("binance.market must be spot or futures")
        intervals = {
            "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
            "6h", "8h", "12h", "1d", "3d", "1w", "1M",
        }
        if self.binance.interval not in intervals:
            raise ConfigError(f"unsupported binance.interval: {self.binance.interval}")
        if self.binance.poll_seconds < 1:
            raise ConfigError("binance.poll_seconds must be at least 1")
        if self.binance.timeout_seconds <= 0:
            raise ConfigError("binance.timeout_seconds must be positive")
        if not 1000 <= self.binance.recv_window <= 60000:
            raise ConfigError("binance.recv_window must be between 1000 and 60000")
        if self.strategy.fast_window < 1:
            raise ConfigError("strategy.fast_window must be positive")
        if self.strategy.slow_window > 500:
            raise ConfigError("strategy.slow_window cannot exceed 500")
        if self.strategy.fast_window >= self.strategy.slow_window:
            raise ConfigError("strategy.fast_window must be smaller than slow_window")
        if self.backtest.initial_quote <= 0:
            raise ConfigError("backtest.initial_quote must be positive")
        if not 0 <= self.backtest.fee_rate < Decimal("0.1"):
            raise ConfigError("backtest.fee_rate must be in [0, 0.1)")
        if not 0 <= self.backtest.slippage_bps <= 1000:
            raise ConfigError("backtest.slippage_bps must be between 0 and 1000")
        if self.risk.max_order_quote <= 0:
            raise ConfigError("risk.max_order_quote must be positive")
        if not 0 < self.risk.max_position_pct <= 1:
            raise ConfigError("risk.max_position_pct must be in (0, 1]")
        if self.risk.max_daily_loss_quote <= 0:
            raise ConfigError("risk.max_daily_loss_quote must be positive")
        if self.risk.cooldown_minutes < 0:
            raise ConfigError("risk.cooldown_minutes cannot be negative")
        if not 1 <= self.futures.leverage <= 20:
            raise ConfigError("futures.leverage must be between 1 and 20")
        if self.futures.margin_type not in {"ISOLATED", "CROSSED"}:
            raise ConfigError("futures.margin_type must be ISOLATED or CROSSED")
        if not str(self.storage.path):
            raise ConfigError("storage.path cannot be empty")


def load_config(path: Path, *, allow_missing: bool = False) -> AppConfig:
    if not path.exists():
        if allow_missing:
            config = AppConfig()
            config.validate()
            return config
        raise ConfigError(f"configuration file not found: {path}")
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from exc

    b = raw.get("binance", {})
    s = raw.get("strategy", {})
    bt = raw.get("backtest", {})
    r = raw.get("risk", {})
    f = raw.get("futures", {})
    st = raw.get("storage", {})
    lv = raw.get("live", {})
    if not all(isinstance(section, dict) for section in (b, s, bt, r, f, st, lv)):
        raise ConfigError("configuration sections must be TOML tables")

    try:
        config = AppConfig(
            binance=BinanceConfig(
                symbol=str(b.get("symbol", "BTCUSDT")).upper(),
                market=str(b.get("market", "spot")).lower(),
                interval=str(b.get("interval", "1h")),
                poll_seconds=_integer(b.get("poll_seconds", 15), "binance.poll_seconds"),
                timeout_seconds=float(b.get("timeout_seconds", 10)),
                recv_window=_integer(b.get("recv_window", 5000), "binance.recv_window"),
            ),
            strategy=StrategyConfig(
                fast_window=_integer(s.get("fast_window", 20), "strategy.fast_window"),
                slow_window=_integer(s.get("slow_window", 50), "strategy.slow_window"),
            ),
            backtest=BacktestConfig(
                initial_quote=_decimal(bt.get("initial_quote", "10000"), "backtest.initial_quote"),
                fee_rate=_decimal(bt.get("fee_rate", "0.001"), "backtest.fee_rate"),
                slippage_bps=_decimal(bt.get("slippage_bps", "5"), "backtest.slippage_bps"),
            ),
            risk=RiskConfig(
                max_order_quote=_decimal(r.get("max_order_quote", "250"), "risk.max_order_quote"),
                max_position_pct=_decimal(r.get("max_position_pct", "0.25"), "risk.max_position_pct"),
                max_daily_loss_quote=_decimal(
                    r.get("max_daily_loss_quote", "100"), "risk.max_daily_loss_quote"
                ),
                cooldown_minutes=_integer(r.get("cooldown_minutes", 0), "risk.cooldown_minutes"),
            ),
            futures=FuturesConfig(
                leverage=_integer(f.get("leverage", 2), "futures.leverage"),
                margin_type=str(f.get("margin_type", "ISOLATED")).upper(),
            ),
            storage=StorageConfig(path=Path(str(st.get("path", "data/binance_auto.db")))),
            live=LiveConfig(enabled=lv.get("enabled", False)),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid configuration: {exc}") from exc
    if not isinstance(config.live.enabled, bool):
        raise ConfigError("live.enabled must be a TOML boolean")
    config.validate()
    return config
