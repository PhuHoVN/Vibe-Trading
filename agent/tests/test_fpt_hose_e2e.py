"""End-to-end HOSE smoke test using FPT.

This test is deliberately offline and deterministic. It exercises the full
daily execution path for the explicit HOSE symbol form:

    HOSE:FPT -> vietnam_equity -> VietnamEquityEngine -> fills

Live SSI/Yahoo connectivity is covered separately by opt-in canaries because
credentials and upstream availability must never decide the normal CI result.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import runner
from backtest.engines._market_hooks import _detect_market
from backtest.engines.vietnam_equity import (
    VietnamEquityEngine,
    hose_tick_size,
)

CODE = "HOSE:FPT"

_BASE = 100_000.0
_BARS = pd.DataFrame(
    {
        "open": [_BASE + 100 * i for i in range(10)],
        "high": [_BASE + 100 * i + 500 for i in range(10)],
        "low": [_BASE + 100 * i - 500 for i in range(10)],
        "close": [_BASE + 100 * i + 100 for i in range(10)],
        "volume": [2_000_000.0] * 10,
    },
    index=pd.bdate_range("2026-09-01", periods=10),
)


class _FakeLoader:
    def fetch(self, *args, **kwargs):
        return {CODE: _BARS.copy()}


class _WeightSignal:
    """Open, hold, then exit; BaseEngine executes target weights one bar later."""

    def generate(self, data_map):
        index = data_map[CODE].index
        weights = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return {CODE: pd.Series(weights, index=index, dtype=float)}


def test_fpt_routes_to_vietnam_equity_engine() -> None:
    assert _detect_market(CODE) == "vietnam_equity"
    # Legacy source is still used only for engine/annualisation compatibility;
    # actual source selection may be SSI through the loader registry.
    assert runner._detect_source(CODE) == "yahoo"


def test_fpt_hose_backtest_executes_with_vietnam_rules(tmp_path: Path) -> None:
    config = {
        "codes": [CODE],
        "start_date": "2026-09-01",
        "end_date": "2026-09-30",
        "source": "auto",
        "interval": "1D",
        "initial_cash": 1_000_000_000.0,
        "slippage": 0.0,
        "position_adjustment": "hold",
    }

    engine = VietnamEquityEngine(config)
    engine.run_backtest(
        config,
        _FakeLoader(),
        _WeightSignal(),
        tmp_path,
    )

    fills = engine.fill_records
    assert fills, "FPT HOSE smoke test must produce at least one fill"
    assert all(fill.symbol == CODE for fill in fills)

    opening = [fill for fill in fills if fill.action == "open"]
    closing = [fill for fill in fills if fill.action == "close"]
    assert opening, "expected an opening FPT fill"
    assert closing, "expected a closing FPT fill after the exit signal"

    # HOSE board-lot rule.
    assert all(abs(fill.signed_quantity) % 100 == 0 for fill in fills)

    # Every execution price must land on the HOSE tick grid.
    for fill in fills:
        tick = hose_tick_size(fill.execution_price)
        assert fill.execution_price % tick == 0

    # Settlement guard: the close cannot happen before two sessions have
    # elapsed from the newest opening fill.
    assert closing[0].bar_idx - opening[-1].bar_idx >= 2
