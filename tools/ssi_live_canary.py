"""Live SSI FastConnect canary for HOSE:FPT.

Read-only by design:
- authenticates only the SSI market-data client;
- fetches daily OHLCV for FPT;
- validates freshness and price caliber;
- runs the returned frame through VietnamEquityEngine.

No Trading client, private key, account number, OTP, or order API is imported.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
import tempfile

import pandas as pd

from backtest.engines.vietnam_equity import VietnamEquityEngine, hose_tick_size
from backtest.loaders.ssi_loader import DataLoader

CODE = "HOSE:FPT"


class _FrameLoader:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def fetch(self, *args, **kwargs):
        return {CODE: self.frame.copy()}


class _Signal:
    def generate(self, data_map):
        idx = data_map[CODE].index
        if len(idx) < 8:
            raise RuntimeError(f"need >=8 bars for engine canary, got {len(idx)}")
        weights = [0.0] * len(idx)
        for i in range(1, min(5, len(idx))):
            weights[i] = 0.5
        return {CODE: pd.Series(weights, index=idx, dtype=float)}


def main() -> None:
    end = date.today()
    start = end - timedelta(days=45)

    loader = DataLoader()
    if not loader.is_available():
        raise RuntimeError(
            "SSI loader unavailable. Configure GitHub Actions secrets "
            "SSI_CLIENT_ID, SSI_API_KEY, SSI_API_SECRET and install the ssi extra."
        )

    data = loader.fetch(
        [CODE],
        start.isoformat(),
        end.isoformat(),
        interval="1D",
    )
    if CODE not in data or data[CODE].empty:
        raise RuntimeError("SSI returned no FPT daily OHLCV data")

    frame = data[CODE].copy()
    latest = pd.Timestamp(frame.index.max()).date()
    age_days = (end - latest).days
    if age_days > 10:
        raise RuntimeError(f"SSI FPT data looks stale: latest bar={latest}, age={age_days} days")

    closes = frame["close"].dropna()
    if closes.empty:
        raise RuntimeError("SSI FPT close series is empty")

    median_close = float(closes.median())
    # FPT on HOSE is quoted in whole VND. A value around 100 rather than
    # ~100,000 would indicate thousand-VND units and would invalidate the
    # engine's tick/lot arithmetic.
    if not 1_000 <= median_close <= 1_000_000:
        raise RuntimeError(
            f"unexpected FPT price caliber from SSI: median close={median_close}; "
            "VietnamEquityEngine expects whole VND"
        )

    config = {
        "codes": [CODE],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "source": "ssi",
        "interval": "1D",
        "initial_cash": 1_000_000_000.0,
        "slippage": 0.0,
        # The canary validates the live-data -> engine path. Corporate-action
        # reference-price enrichment is a separate phase, so disable the band
        # here rather than infer a possibly adjusted reference price.
        "price_limit": 0,
        "position_adjustment": "hold",
    }

    with tempfile.TemporaryDirectory(prefix="ssi-fpt-canary-") as tmp:
        engine = VietnamEquityEngine(config)
        engine.run_backtest(
            config,
            _FrameLoader(frame),
            _Signal(),
            Path(tmp),
        )

    fills = engine.fill_records
    if not fills:
        raise RuntimeError("VietnamEquityEngine produced no FPT fills from live SSI data")

    for fill in fills:
        if abs(fill.signed_quantity) % 100 != 0:
            raise RuntimeError(f"non-HOSE board-lot fill: {fill.signed_quantity}")
        tick = hose_tick_size(fill.execution_price)
        if fill.execution_price % tick != 0:
            raise RuntimeError(
                f"off-grid HOSE fill price: {fill.execution_price} (tick={tick})"
            )

    summary = {
        "status": "PASS",
        "symbol": CODE,
        "bars": len(frame),
        "first_bar": str(pd.Timestamp(frame.index.min()).date()),
        "latest_bar": str(latest),
        "latest_close_vnd": float(closes.iloc[-1]),
        "median_close_vnd": median_close,
        "fills": len(fills),
        "first_fill_bar": int(fills[0].bar_idx),
        "last_fill_bar": int(fills[-1].bar_idx),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
