#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Golden Triangle (有效买托) Backtest Runner
============================================

Run an event-driven backtest of the Golden Triangle selector using Qlib's
backtest engine.

Examples::

    # Use Qlib local data with turnover already imported
    python run_backtest.py \
        --provider-uri ~/.qlib/qlib_data/cn_data \
        --turnover-source qlib \
        --start 2024-01-01 \
        --end 2024-12-31 \
        --max-positions 10 \
        --holding-period 5 \
        --account 1000000

    # Use Qlib OHLCV + Tushare historical turnover (bulk by trade_date, faster)
    python run_backtest.py \
        --provider-uri ~/.qlib/qlib_data/cn_data \
        --turnover-source tushare \
        --tushare-token your_token \
        --start 2024-01-01 \
        --end 2024-06-30 \
        --instruments csi300

    # Use Qlib OHLCV + akshare historical turnover
    python run_backtest.py \
        --provider-uri ~/.qlib/qlib_data/cn_data \
        --turnover-source akshare_hist \
        --start 2024-01-01 \
        --end 2024-06-30 \
        --instruments csi300
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import qlib
from qlib.constant import REG_CN
from qlib.backtest import backtest
from qlib.data import D
from qlib.contrib.golden_triangle.strategy import GoldenTriangleStrategy
from qlib.contrib.golden_triangle.data_source import HybridDataSource


def parse_args():
    p = argparse.ArgumentParser(description="Golden Triangle Backtest")
    p.add_argument("--provider-uri", default=None, help="Qlib data directory")
    p.add_argument("--region", default="cn", help="Qlib region")
    p.add_argument("--instruments", default="csi300", help="Stock pool: all, csi300, csi500, or comma-separated list")
    p.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
    p.add_argument("--account", type=float, default=1_000_000, help="Initial cash")
    p.add_argument("--benchmark", default="SH000300", help="Benchmark code")
    p.add_argument("--max-positions", type=int, default=10, help="Max simultaneous holdings")
    p.add_argument("--holding-period", type=int, default=5, help="Holding period in trading days")
    p.add_argument("--sell-on-exit-signal", action="store_true", help="Sell when a stock falls out of selection")
    p.add_argument("--lookback", type=int, default=35, help="Selector lookback days")
    p.add_argument("--obs-window", type=int, default=3, help="Golden triangle observation window")
    p.add_argument("--volume-lookback", type=int, default=5, help="Volume average lookback")
    p.add_argument("--volume-multiplier", type=float, default=1.5, help="Volume surge threshold")
    p.add_argument("--turnover-threshold", type=float, default=3.0, help="Min turnover %%")
    p.add_argument("--turnover-source", default="qlib", choices=["qlib", "akshare_spot", "akshare_hist", "tushare", "csv"], help="Turnover source. Recommended: pre-import turnover into Qlib and use 'qlib' for backtest speed.")
    p.add_argument("--turnover-csv", default=None, help="Local turnover CSV path")
    p.add_argument("--tushare-token", default=None, help="Tushare token (or set TUSHARE_TOKEN env var)")
    p.add_argument("--tushare-delay", type=float, default=0.3, help="Seconds to sleep between Tushare API calls")
    p.add_argument("--industry-filter", default=None, help="Comma-separated industry keywords")
    p.add_argument("--no-st-filter", action="store_true", help="Do not filter ST stocks")
    p.add_argument("--open-cost", type=float, default=0.0005, help="Open commission")
    p.add_argument("--close-cost", type=float, default=0.0015, help="Close commission")
    p.add_argument("--limit-threshold", type=float, default=0.095, help="Limit up/down threshold")
    p.add_argument("--deal-price", default="open", help="Execution price: open, close, vwap")
    p.add_argument("--output-dir", default="examples/golden_triangle/backtest_output", help="Output directory")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def resolve_instruments(arg: str):
    if "," in arg:
        return [c.strip() for c in arg.split(",")]
    return arg


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    # Initialize Qlib.  The backtest engine needs local price/volume data.
    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    instruments = resolve_instruments(args.instruments)
    turnover_kwargs = {}
    if args.turnover_source == "csv":
        if not args.turnover_csv:
            raise ValueError("--turnover-csv is required when --turnover-source=csv")
        turnover_kwargs["path"] = args.turnover_csv
    if args.tushare_token:
        turnover_kwargs["api_token"] = args.tushare_token
    turnover_kwargs["delay"] = args.tushare_delay

    industry_filter = [k.strip() for k in args.industry_filter.split(",")] if args.industry_filter else None

    strategy = GoldenTriangleStrategy(
        provider_uri=args.provider_uri,
        region=args.region,
        turnover_source=args.turnover_source,
        turnover_kwargs=turnover_kwargs,
        max_positions=args.max_positions,
        holding_period=args.holding_period,
        sell_on_exit_signal=args.sell_on_exit_signal,
        lookback_days=args.lookback,
        observation_window=args.obs_window,
        volume_lookback=args.volume_lookback,
        volume_multiplier=args.volume_multiplier,
        turnover_threshold=args.turnover_threshold,
        filter_st=not args.no_st_filter,
        industry_filter=industry_filter,
    )

    # Pre-fetch static info (stock names/industries, ST list) to fail fast.
    strategy._prepare_static_info()

    exchange_kwargs = {
        "freq": "day",
        "limit_threshold": args.limit_threshold,
        "deal_price": args.deal_price,
        "open_cost": args.open_cost,
        "close_cost": args.close_cost,
        "min_cost": 5,
    }

    # Qlib data directories typically store symbols in lower case (e.g. sh000300).
    # Normalize the benchmark code so that user inputs like "SH000300" still work.
    benchmark = args.benchmark.lower()

    # Validate that local Qlib data covers the requested backtest window.
    # This produces a much clearer error than the deep stack from backtest().
    _codes_to_check = [benchmark]
    if isinstance(instruments, list):
        # Individual stock codes; spot-check a few of them.
        _codes_to_check.extend(instruments[:3])
    for code in _codes_to_check:
        code = code.lower() if isinstance(code, str) else code
        try:
            coverage_df = D.features([code], ["$close"], start_time=args.start, end_time=args.end, freq="day")
        except Exception as e:
            logger.warning(f"Could not validate coverage for '{code}': {e}")
            continue
        if coverage_df.empty:
            try:
                all_df = D.features([code], ["$close"], freq="day")
            except Exception as e:
                all_df = pd.DataFrame()
            if all_df.empty:
                raise ValueError(
                    f"No local Qlib data found for instrument '{code}'. "
                    f"Please check --provider-uri and the data directory."
                )
            min_date = all_df.index.get_level_values("datetime").min().strftime("%Y-%m-%d")
            max_date = all_df.index.get_level_values("datetime").max().strftime("%Y-%m-%d")
            raise ValueError(
                f"Instrument '{code}' has no data for {args.start} ~ {args.end}. "
                f"Local data range: {min_date} ~ {max_date}. "
                f"Please adjust --start/--end or download newer Qlib data."
            )

    logger.info(f"Running backtest {args.start} ~ {args.end} on {args.instruments}")
    portfolio_metric, indicator_metric = backtest(
        start_time=args.start,
        end_time=args.end,
        strategy=strategy,
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        account=args.account,
        benchmark=benchmark,
        exchange_kwargs=exchange_kwargs,
        pos_type="Position",
    )

    # Save results.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Portfolio metrics (account value, cash, return, etc.)
    if isinstance(portfolio_metric, dict):
        # Convert nested dict of Account to DataFrame.
        records = []
        for account_id, metrics_df in portfolio_metric.items():
            if metrics_df is not None and not metrics_df.empty:
                records.append(metrics_df)
        if records:
            port_df = pd.concat(records)
        else:
            port_df = pd.DataFrame()
    else:
        port_df = portfolio_metric if isinstance(portfolio_metric, pd.DataFrame) else pd.DataFrame()

    if not port_df.empty:
        port_path = out_dir / "portfolio_metrics.csv"
        port_df.to_csv(port_path, encoding="utf-8-sig")
        logger.info(f"Portfolio metrics saved: {port_path}")
        print("\n" + "=" * 60)
        print("回测组合净值（部分）")
        print("=" * 60)
        print(port_df.head(10).to_string())
        print("...")
        print(port_df.tail(10).to_string())

    # Indicator metrics (annual return, sharpe, max drawdown, etc.)
    ind_path = out_dir / "indicator_metrics.csv"
    if isinstance(indicator_metric, dict):
        rows = []
        for k, v in indicator_metric.items():
            if hasattr(v, "to_frame"):
                rows.append(v.to_frame().T)
            else:
                rows.append(pd.DataFrame([{"metric": k, "value": v}]))
        if rows:
            ind_df = pd.concat(rows, ignore_index=True)
        else:
            ind_df = pd.DataFrame()
    else:
        ind_df = indicator_metric if isinstance(indicator_metric, pd.DataFrame) else pd.DataFrame()

    if not ind_df.empty:
        ind_df.to_csv(ind_path, encoding="utf-8-sig")
        logger.info(f"Indicator metrics saved: {ind_path}")
        print("\n" + "=" * 60)
        print("绩效指标")
        print("=" * 60)
        print(ind_df.to_string(index=False))

    print(f"\n回测完成，结果保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
