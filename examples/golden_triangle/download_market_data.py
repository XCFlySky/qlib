#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载全市场 OHLCV + 换手率数据，保存为 CSV 供 optimize_params.py 使用。
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def fetch_all_from_tushare(start_date: str, end_date: str, token: str = None, delay: float = 0.3):
    try:
        import tushare as ts
    except ImportError:
        raise ImportError("tushare is required. pip install tushare")

    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("Tushare token required. Pass --token or set TUSHARE_TOKEN env var.")

    pro = ts.pro_api(token)

    # 交易日历
    cal = pro.trade_cal(
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        is_open="1",
    )
    trade_dates = sorted(cal["cal_date"].tolist())
    logger.info(f"Total trading days: {len(trade_dates)}")

    records = []
    for idx, td in enumerate(trade_dates, 1):
        try:
            df_daily = pro.daily(trade_date=td)
            df_basic = pro.daily_basic(trade_date=td)
            if df_daily is None or df_daily.empty:
                continue
            if df_basic is not None and not df_basic.empty:
                df = df_daily.merge(
                    df_basic[["ts_code", "trade_date", "turnover_rate"]],
                    on=["ts_code", "trade_date"],
                    how="left",
                )
            else:
                df = df_daily.copy()
                df["turnover_rate"] = np.nan

            df = df.rename(columns={
                "trade_date": "datetime",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "turnover_rate": "turnover",
            })
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["instrument"] = df["ts_code"].apply(
                lambda x: f"SH{x[:6]}" if x.endswith(".SH") else f"SZ{x[:6]}"
            )
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100
            for col in ["open", "high", "low", "close", "turnover"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            records.append(df[["datetime", "instrument", "open", "high", "low", "close", "volume", "turnover"]])
        except Exception as e:
            logger.warning(f"Tushare failed for {td}: {e}")

        if idx % 50 == 0:
            logger.info(f"... processed {idx}/{len(trade_dates)} days")
        time.sleep(delay)

    if not records:
        raise RuntimeError("No data fetched.")
    df = pd.concat(records, ignore_index=True)
    df = df.set_index(["datetime", "instrument"]).sort_index()
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-06-30")
    p.add_argument("--token", default=None, help="Tushare token")
    p.add_argument("--output", default="data/cn_data_2020_2024.csv")
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    df = fetch_all_from_tushare(args.start, args.end, args.token)
    logger.info(f"Fetched {len(df)} rows, {df.index.get_level_values('instrument').nunique()} instruments")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, encoding="utf-8-sig")
    logger.info(f"Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
