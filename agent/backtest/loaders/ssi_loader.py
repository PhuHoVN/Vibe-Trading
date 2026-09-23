"""SSI FastConnect V3 market-data loader for Vietnam equities.

The loader is deliberately read-only: it uses SSI's official Python SDK only
for historical OHLCV and never imports or calls trading/order APIs.

Install with: pip install -e ".[ssi]"

Credentials are read from SSI_CLIENT_ID, SSI_API_KEY and SSI_API_SECRET.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.loaders.base import cached_loader_fetch, validate_date_range, validate_ohlc
from backtest.loaders.registry import register
from src.config.accessor import get_env_value

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_DAILY_ALIASES = {"1d", "d", "day", "daily"}


def map_symbol(code: str) -> str:
    """Map a Vibe-Trading Vietnam symbol to SSI's bare ticker convention."""
    symbol = str(code).strip().upper()
    if symbol.endswith(".VN"):
        symbol = symbol[:-3]
    if ":" in symbol:
        venue, bare = symbol.split(":", 1)
        if venue in {"HOSE", "HNX", "UPCOM"} and bare:
            symbol = bare
    return symbol


def _credentials() -> tuple[str, str, str]:
    """Return SSI client id, API key and API secret from the config layer."""
    return (
        get_env_value("SSI_CLIENT_ID", "").strip(),
        get_env_value("SSI_API_KEY", "").strip(),
        get_env_value("SSI_API_SECRET", "").strip(),
    )


def _load_sdk() -> tuple[Any, Any, Any] | None:
    """Lazy-import the optional SSI SDK."""
    try:
        from ssi_sdk import Auth, Config, Data
    except ImportError:
        return None
    return Auth, Config, Data


def _field(item: Any, *names: str) -> Any:
    """Read the first matching field from an SDK model or dict."""
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _normalize(candles: Any) -> Optional[pd.DataFrame]:
    """Normalize SSI SDK OHLC models into the project OHLCV contract."""
    if candles is None:
        return None

    rows: list[dict[str, Any]] = []
    for item in candles:
        row = {
            "trade_date": _field(item, "trading_date", "tradingDate"),
            "open": _field(item, "open_price", "openPrice"),
            "high": _field(item, "high_price", "highPrice"),
            "low": _field(item, "low_price", "lowPrice"),
            "close": _field(item, "close_price", "closePrice"),
            "volume": _field(item, "volume"),
        }
        if row["trade_date"] is None or any(
            row[key] is None for key in ("open", "high", "low", "close")
        ):
            continue
        rows.append(row)

    if not rows:
        return None

    frame = pd.DataFrame(rows)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in _OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = (
        frame.dropna(subset=["trade_date", "open", "high", "low", "close"])
        .set_index("trade_date")
        .sort_index()
    )
    if frame.empty:
        return None

    frame.index = pd.DatetimeIndex(frame.index).astype("datetime64[ns]")
    frame.index.name = "trade_date"
    frame = frame[_OHLCV_COLUMNS].astype(float)
    frame = validate_ohlc(frame, strategy="drop")
    if frame.empty:
        return None
    frame.attrs["quote_currency"] = "VND"
    return frame


def _ssi_date(date_str: str, *, end_of_day: bool) -> str:
    """Convert YYYY-MM-DD into the FastConnect SDK historical-date format."""
    day = pd.Timestamp(date_str).strftime("%Y/%m/%d")
    return f"{day} {'23:59:59' if end_of_day else '00:00:00'}"


@register
class DataLoader:
    """Read-only SSI FastConnect V3 daily OHLCV loader."""

    name = "ssi"
    markets = {"vietnam_equity"}
    requires_auth = True

    def is_available(self) -> bool:
        """Return whether both the optional SDK and all market-data keys exist."""
        client_id, api_key, api_secret = _credentials()
        return bool(client_id and api_key and api_secret and _load_sdk() is not None)

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical daily OHLCV keyed by the original input symbols."""
        del fields
        validate_date_range(start_date, end_date)

        if str(interval).strip().lower() not in _DAILY_ALIASES:
            logger.warning("ssi loader currently serves daily bars only; interval=%r", interval)
            return {}

        sdk = _load_sdk()
        client_id, api_key, api_secret = _credentials()
        if sdk is None or not (client_id and api_key and api_secret):
            logger.warning(
                "ssi fetch skipped: install .[ssi] and set SSI_CLIENT_ID, "
                "SSI_API_KEY and SSI_API_SECRET"
            )
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            try:
                frame = cached_loader_fetch(
                    source=self.name,
                    symbol=code,
                    timeframe=interval,
                    start_date=start_date,
                    end_date=end_date,
                    fields=None,
                    fetch=lambda code=code: self._fetch_one(
                        code, start_date, end_date, sdk, client_id, api_key, api_secret
                    ),
                )
                if frame is not None and not frame.empty:
                    result[code] = frame
            except Exception as exc:
                logger.warning("ssi failed for %s: %s", code, exc)
        return result

    @staticmethod
    def _fetch_one(
        code: str,
        start_date: str,
        end_date: str,
        sdk: tuple[Any, Any, Any],
        client_id: str,
        api_key: str,
        api_secret: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch one symbol through the official SSI SDK and normalize it."""
        Auth, Config, Data = sdk
        config = Config(client_id=client_id, api_key=api_key, api_secret=api_secret)
        with Auth(config) as auth:
            auth.authenticate()
            with Data(auth) as data:
                candles = data.market_data.get_ohlc_1day_historical(
                    map_symbol(code),
                    from_date=_ssi_date(start_date, end_of_day=False),
                    to_date=_ssi_date(end_date, end_of_day=True),
                )
        return _normalize(candles)
