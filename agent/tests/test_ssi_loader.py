"""Unit tests for the SSI FastConnect V3 loader.

The official SDK is fully faked here; tests never call SSI over the network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from backtest.loaders import ssi_loader as mod
from backtest.loaders.registry import FALLBACK_CHAINS, is_no_network_fallback_source
from backtest.loaders.ssi_loader import DataLoader, _is_hose_symbol, _normalize, map_symbol


def _set_credentials(monkeypatch) -> None:
    monkeypatch.setenv("SSI_CLIENT_ID", "client")
    monkeypatch.setenv("SSI_API_KEY", "api-key")
    monkeypatch.setenv("SSI_API_SECRET", "api-secret")
    monkeypatch.setenv("VIBE_TRADING_DATA_CACHE", "0")


def _candles():
    return [
        SimpleNamespace(
            trading_date="2026-09-22",
            open_price=100.0,
            high_price=105.0,
            low_price=99.0,
            close_price=104.0,
            volume=1_000,
        ),
        SimpleNamespace(
            trading_date="2026-09-23",
            open_price=104.0,
            high_price=107.0,
            low_price=103.0,
            close_price=106.0,
            volume=2_000,
        ),
    ]


def test_symbol_mapping() -> None:
    assert map_symbol("FPT.VN") == "FPT"
    assert map_symbol("hose:fpt") == "FPT"
    assert map_symbol("HNX:SHS") == "SHS"
    assert map_symbol("UPCOM:ACV") == "ACV"
    assert _is_hose_symbol("FPT.VN") is True
    assert _is_hose_symbol("HOSE:FPT") is True
    assert _is_hose_symbol("HNX:SHS") is False
    assert _is_hose_symbol("UPCOM:ACV") is False


def test_normalize_sdk_models() -> None:
    frame = _normalize(_candles())
    assert frame is not None
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.dtype == "datetime64[ns]"
    assert frame.index.name == "trade_date"
    assert frame.attrs["quote_currency"] == "VND"
    assert frame["close"].tolist() == [104.0, 106.0]


def test_normalize_accepts_dict_payloads() -> None:
    frame = _normalize(
        [{
            "tradingDate": "2026-09-23",
            "openPrice": 10,
            "highPrice": 12,
            "lowPrice": 9,
            "closePrice": 11,
            "volume": 500,
        }]
    )
    assert frame is not None
    assert frame.iloc[0]["close"] == 11.0


def test_unavailable_without_credentials(monkeypatch) -> None:
    for name in ("SSI_CLIENT_ID", "SSI_API_KEY", "SSI_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mod, "_load_sdk", lambda: (object, object, object))
    assert DataLoader().is_available() is False


def test_available_with_sdk_and_credentials(monkeypatch) -> None:
    _set_credentials(monkeypatch)
    monkeypatch.setattr(mod, "_load_sdk", lambda: (object, object, object))
    assert DataLoader().is_available() is True


def test_fetch_daily_with_fake_sdk(monkeypatch) -> None:
    _set_credentials(monkeypatch)
    calls = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

    class FakeAuth:
        def __init__(self, config):
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def authenticate(self):
            calls["authenticated"] = True

    class FakeMarketData:
        def get_ohlc_1day_historical(self, symbol, *, from_date, to_date, page, size):
            calls["ohlc"] = (symbol, from_date, to_date, page, size)
            return _candles()

    class FakeData:
        def __init__(self, auth):
            self.market_data = FakeMarketData()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(mod, "_load_sdk", lambda: (FakeAuth, FakeConfig, FakeData))

    out = DataLoader().fetch(["FPT.VN"], "2026-09-01", "2026-09-23")

    assert list(out) == ["FPT.VN"]
    assert calls["authenticated"] is True
    assert calls["ohlc"] == (
        "FPT",
        "2026/09/01 00:00:00",
        "2026/09/23 23:59:59",
        1,
        1000,
    )
    assert out["FPT.VN"]["volume"].tolist() == [1000.0, 2000.0]


def test_fetch_explicit_hose_symbol_with_fake_sdk(monkeypatch) -> None:
    _set_credentials(monkeypatch)
    calls = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

    class FakeAuth:
        def __init__(self, config):
            self.config = config
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def authenticate(self):
            pass

    class FakeMarketData:
        def get_ohlc_1day_historical(self, symbol, *, from_date, to_date, page, size):
            calls["symbol"] = symbol
            calls["page_size"] = (page, size)
            return _candles()

    class FakeData:
        def __init__(self, auth):
            self.market_data = FakeMarketData()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(mod, "_load_sdk", lambda: (FakeAuth, FakeConfig, FakeData))
    out = DataLoader().fetch(["HOSE:FPT"], "2026-09-01", "2026-09-23")
    assert list(out) == ["HOSE:FPT"]
    assert calls["symbol"] == "FPT"
    assert calls["page_size"] == (1, 1000)


def test_hnx_and_upcom_are_rejected_in_hose_phase(monkeypatch) -> None:
    _set_credentials(monkeypatch)
    monkeypatch.setattr(mod, "_load_sdk", lambda: (object, object, object))
    assert DataLoader().fetch(["HNX:SHS", "UPCOM:ACV"], "2026-09-01", "2026-09-23") == {}


def test_non_daily_interval_is_rejected(monkeypatch) -> None:
    _set_credentials(monkeypatch)
    assert DataLoader().fetch(["FPT.VN"], "2026-09-01", "2026-09-23", interval="1H") == {}


def test_registry_prefers_ssi_for_vietnam() -> None:
    assert FALLBACK_CHAINS["vietnam_equity"][0] == "ssi"
    assert is_no_network_fallback_source("ssi") is True
