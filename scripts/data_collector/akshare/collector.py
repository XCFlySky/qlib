#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
AKShare Turnover Collector for Qlib
====================================
Collects historical turnover (换手率) and optionally OHLCV for all
A-share stocks via akshare, normalises them to Qlib CSV format, and
dumps them into Qlib binary storage so that ``$turnover`` can be read
natively by ``D.features()``.

Usage::

    # Full history (slow – one request per stock)
    python collector.py --start 2020-01-01 --end 2024-12-31 \
                        --output-dir ~/.qlib/akshare_csv \
                        --qlib-dir ~/.qlib/qlib_data/cn_data

    # Incremental update (last 30 days)
    python collector.py --incremental-days 30 \
                        --output-dir ~/.qlib/akshare_csv \
                        --qlib-dir ~/.qlib/qlib_data/cn_data

    # Only fetch turnover CSV, skip dump_bin
    python collector.py --start 2024-01-01 --end 2024-12-31 \
                        --output-dir ~/.qlib/akshare_csv \
                        --skip-dump

"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# Ensure repo root is on path so we can import dump_bin
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.dump_bin import DumpDataUpdate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--incremental-days", type=int, default=None, help="If set, override start/end to last N days")
    p.add_argument("--output-dir", required=True, help="Directory to save normalised CSV files")
    p.add_argument("--qlib-dir", default=None, help="Qlib data directory for dump_bin (e.g. ~/.qlib/qlib_data/cn_data)")
    p.add_argument("--skip-dump", action="store_true", help="Skip dump_bin.py, only generate CSVs")
    p.add_argument("--delay", type=float, default=0.3, help="Seconds to sleep between akshare requests")
    p.add_argument("--limit", type=int, default=None, help="Limit number of stocks (for testing)")
    p.add_argument("--fields", default="turnover", help="Comma-separated fields to collect: turnover,ohlcv")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def qlib_code(symbol: str) -> Optional[str]:
    """600000 -> SH600000"""
    s = str(symbol).strip()
    if len(s) == 6 and s.isdigit():
        return f"SH{s}" if s.startswith("6") else f"SZ{s}"
    return None


def fetch_stock_list() -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare is required. pip install akshare")
    logger.info("Fetching stock list from akshare ...")
    df = ak.stock_zh_a_spot_em()
    df = df.rename(columns={"代码": "code", "名称": "name"})
    df["qlib_code"] = df["code"].apply(qlib_code)
    df = df.dropna(subset=["qlib_code"])
    return df[["code", "name", "qlib_code"]]


def normalise_to_qlib(df: pd.DataFrame, inst: str) -> pd.DataFrame:
    """
    Convert akshare ``stock_zh_a_hist`` output to Qlib CSV schema:
    ``date, symbol, $close, $volume, $turnover, ...``
    """
    df = df.copy()
    # akshare columns (中文):
    # 日期 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
    col_map = {
        "日期": "date",
        "开盘": "$open",
        "收盘": "$close",
        "最高": "$high",
        "最低": "$low",
        "成交量": "$volume",
        "成交额": "$amount",
        "换手率": "$turnover",
    }
    for old, new in col_map.items():
        if old in df.columns:
            df[new] = df[old]

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = inst

    # Ensure numeric
    for c in ["$open", "$close", "$high", "$low", "$volume", "$turnover"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Qlib CSV order
    out_cols = ["date", "symbol", "$open", "$close", "$high", "$low", "$volume", "$turnover"]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols]


def run():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare not installed. Run: pip install akshare")
        sys.exit(1)

    # ---- date range --------------------------------------------------
    if args.incremental_days:
        end = datetime.now().date()
        start = end - timedelta(days=args.incremental_days)
        args.end = end.strftime("%Y-%m-%d")
        args.start = start.strftime("%Y-%m-%d")
    else:
        if args.end is None:
            args.end = datetime.now().strftime("%Y-%m-%d")
        if args.start is None:
            args.start = (pd.Timestamp(args.end) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    ak_start = args.start.replace("-", "")
    ak_end = args.end.replace("-", "")
    logger.info(f"Date range: {args.start} ~ {args.end}")

    # ---- stock list --------------------------------------------------
    stocks = fetch_stock_list()
    if args.limit:
        stocks = stocks.head(args.limit)
    logger.info(f"Stocks to collect: {len(stocks)}")

    # ---- output dir --------------------------------------------------
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    fields_to_collect = [f.strip().lower() for f in args.fields.split(",")]
    collect_ohlcv = "ohlcv" in fields_to_collect
    collect_turnover = "turnover" in fields_to_collect

    # ---- download & normalise ----------------------------------------
    success = 0
    for idx, row in stocks.iterrows():
        inst = row["qlib_code"]
        symbol = row["code"]
        csv_path = out_dir / f"{inst}.csv"

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=ak_start,
                end_date=ak_end,
                adjust="qfq",
            )
            if df is None or df.empty:
                continue

            norm = normalise_to_qlib(df, inst)

            # If user only wants turnover, drop price columns to save space
            if not collect_ohlcv:
                drop_cols = [c for c in ["$open", "$close", "$high", "$low", "$volume"] if c in norm.columns]
                norm = norm.drop(columns=drop_cols)
            if not collect_turnover and "$turnover" in norm.columns:
                norm = norm.drop(columns=["$turnover"])

            if len(norm.columns) <= 2:  # only date + symbol
                continue

            # Merge with existing CSV if doing incremental update
            if csv_path.exists():
                old = pd.read_csv(csv_path)
                old["date"] = pd.to_datetime(old["date"])
                norm = pd.concat([old, norm], ignore_index=True)
                norm = norm.drop_duplicates(subset=["date", "symbol"], keep="last")
                norm = norm.sort_values("date")

            norm.to_csv(csv_path, index=False)
            success += 1
        except Exception as e:
            logger.debug(f"Failed {inst} ({symbol}): {e}")

        if (idx + 1) % 100 == 0:
            logger.info(f"... collected {idx + 1}/{len(stocks)} stocks")
        time.sleep(args.delay)

    logger.info(f"CSV generation complete. Success: {success}/{len(stocks)}. Files in: {out_dir}")

    # ---- dump to qlib binary -----------------------------------------
    if not args.skip_dump:
        if not args.qlib_dir:
            logger.error("--qlib-dir is required for dump_bin. Use --skip-dump to skip.")
            sys.exit(1)

        logger.info("Running dump_bin.py update ...")
        include_fields = ""
        if collect_turnover and not collect_ohlcv:
            include_fields = "$turnover"

        dumper = DumpDataUpdate(
            data_path=str(out_dir),
            qlib_dir=str(Path(args.qlib_dir).expanduser()),
            freq="day",
            include_fields=include_fields,
        )
        dumper.dump()
        logger.info("dump_bin.py finished.")
    else:
        logger.info("Skipped dump_bin (--skip-dump).")


if __name__ == "__main__":
    run()
