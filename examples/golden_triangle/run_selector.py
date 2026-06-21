#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Golden Triangle (有效买托) Stock Selector – Runner
====================================================

Usage::

    # 1) Full market scan using Qlib + akshare for turnover
    python run_selector.py --provider-uri ~/.qlib/qlib_data/cn_data \
                           --turnover-source akshare_hist \
                           --output result.csv

    # 2) Scan a specific index pool
    python run_selector.py --instruments csi300 --output result.xlsx

    # 3) Fallback: fetch everything from akshare (no Qlib data needed)
    python run_selector.py --fallback-akshare --output result.csv

Environment variables::

    TUSHARE_TOKEN   – Required when ``--turnover-source tushare``.

"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# Allow import from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qlib.contrib.golden_triangle import HybridDataSource
from qlib.contrib.golden_triangle.enhanced_selector import EnhancedGoldenTriangleSelector
from qlib.contrib.golden_triangle.macro_check import MacroCheck


def parse_args():
    p = argparse.ArgumentParser(description="Golden Triangle Stock Selector")
    p.add_argument("--provider-uri", default=None, help="Qlib data directory")
    p.add_argument("--instruments", default="all", help="Instrument pool: all, csi300, csi500, or comma-separated list")
    p.add_argument("--trade-date", default=None, help="Observation anchor date (YYYY-MM-DD). Default: latest available.")
    p.add_argument("--lookback", type=int, default=35, help="Lookback days (must cover MA20 + 7-day volume avg + obs window)")
    p.add_argument("--obs-window", type=int, default=3, help="Observation window (T, T-1, T-2)")
    p.add_argument("--volume-lookback", type=int, default=5, help="Volume comparison lookback days (default 5, i.e. compare vs prior 5-day avg)")
    p.add_argument("--volume-multiplier", type=float, default=1.5, help="Volume surge threshold")
    p.add_argument("--turnover-threshold", type=float, default=3.0, help="Min turnover rate %%")
    p.add_argument("--predict-ma-ratio", type=float, default=0.985, help="Predicting state min MA10/MA20 ratio")
    p.add_argument("--predict-vol-ratio", type=float, default=1.3, help="Predicting state min volume ratio")
    p.add_argument("--predict-turnover", type=float, default=2.5, help="Predicting state min turnover %%")
    p.add_argument("--forming-ma-ratio", type=float, default=0.97, help="Forming state min MA10/MA20 ratio")
    p.add_argument("--forming-vol-ratio", type=float, default=1.1, help="Forming state min volume ratio")
    p.add_argument("--forming-turnover", type=float, default=2.0, help="Forming state min turnover %%")
    p.add_argument("--macro-check", action="store_true", help="Check overnight US/Asia market performance via tushare before screening")
    p.add_argument("--turnover-source", default="akshare_hist", choices=["qlib", "akshare_spot", "akshare_hist", "tushare", "csv"], help="Where to get turnover data")
    p.add_argument("--turnover-csv", default=None, help="Path to local turnover CSV (required when turnover-source=csv)")
    p.add_argument("--tushare-token", default=None, help="Tushare pro token (or set TUSHARE_TOKEN env var)")
    p.add_argument("--fallback-akshare", action="store_true", help="Use akshare for OHLCV as well (no Qlib data needed)")
    p.add_argument("--fallback-tushare", action="store_true", help="Use tushare pro for OHLCV + turnover (no Qlib data needed)")
    p.add_argument("--industry-filter", default=None, help="Comma-separated industry keywords to keep (optional)")
    p.add_argument("--output", default=None, help="Output base name. Result will be saved to examples/golden_triangle/output/<name>_YYYYMMDD.csv/.xlsx. Default: result")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def resolve_instruments(arg: str):
    if "," in arg:
        return [c.strip() for c in arg.split(",")]
    return arg


def fetch_fallback_tushare(instruments, start_date: str, end_date: str, api_token: str = None):
    """
    Fetch OHLCV + turnover from Tushare Pro via trade_date-based bulk queries.
    Much faster than per-stock fetching (only ~N_days API calls).
    """
    import time
    try:
        import tushare as ts
    except ImportError:
        raise ImportError("tushare is required. pip install tushare")

    if api_token is None:
        api_token = os.environ.get("TUSHARE_TOKEN")
    if not api_token:
        raise ValueError("Tushare API token is required. Pass --tushare-token or set TUSHARE_TOKEN env var.")

    pro = ts.pro_api(api_token)

    # Resolve trading calendar
    logger.info("Fetching trading calendar from tushare ...")
    cal = pro.trade_cal(
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        is_open="1",
    )
    if cal is None or cal.empty:
        raise RuntimeError("Tushare returned empty trade calendar.")
    trade_dates = sorted(cal["cal_date"].tolist())
    logger.info(f"Total trading days in range: {len(trade_dates)}")

    records = []
    for td in trade_dates:
        try:
            # OHLCV
            df_daily = pro.daily(trade_date=td)
            if df_daily is None or df_daily.empty:
                continue

            # Turnover
            df_basic = pro.daily_basic(trade_date=td)
            if df_basic is not None and not df_basic.empty:
                df = df_daily.merge(
                    df_basic[["ts_code", "trade_date", "turnover_rate"]],
                    on=["ts_code", "trade_date"],
                    how="left",
                )
            else:
                df = df_daily.copy()
                df["turnover_rate"] = None

            # Normalise
            df = df.rename(columns={
                "trade_date": "datetime",
                "close": "close",
                "vol": "volume",
                "turnover_rate": "turnover",
            })
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["instrument"] = df["ts_code"].apply(
                lambda x: f"SH{x[:6]}" if x.endswith(".SH") else f"SZ{x[:6]}"
            )

            # Filter to requested instruments
            if isinstance(instruments, list):
                df = df[df["instrument"].isin(instruments)]

            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100  # 手 -> 股
            df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")

            records.append(df[["datetime", "instrument", "close", "volume", "turnover"]])
        except Exception as e:
            logger.warning(f"Tushare fetch failed for {td}: {e}")
        time.sleep(0.3)  # rate limit

    if not records:
        raise RuntimeError("No data fetched from tushare fallback.")

    df = pd.concat(records, ignore_index=True)
    df = df.set_index(["datetime", "instrument"]).sort_index()
    logger.info(f"Tushare data loaded: {len(df)} rows, {df.index.get_level_values('instrument').nunique()} stocks")

    # Fetch stock name + industry
    logger.info("Fetching stock name/industry from tushare ...")
    try:
        stock_basic = pro.stock_basic(exchange="", list_status="L")
        stock_basic["instrument"] = stock_basic["ts_code"].apply(
            lambda x: f"SH{x[:6]}" if x.endswith(".SH") else f"SZ{x[:6]}"
        )
        stock_info = stock_basic.set_index("instrument")[["name", "industry"]]
    except Exception as e:
        logger.warning(f"Could not fetch stock info from tushare: {e}")
        stock_info = None

    return df, stock_info


def fetch_fallback_akshare(instruments: list, start_date: str, end_date: str, delay: float = 0.5):
    """
    When user has no Qlib data, fetch OHLCV + turnover entirely from akshare.
    This is slower but requires zero local binary data.
    """
    import time
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare is required for --fallback-akshare. pip install akshare")

    records = []
    failed = []
    logger.info(f"Fallback mode: fetching OHLCV+turnover from akshare for {len(instruments)} stocks ...")
    for idx, inst in enumerate(instruments, 1):
        symbol = inst[-6:]
        success = False
        for attempt in range(3):  # retry up to 3 times
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )
                if df is None or df.empty:
                    break
                df = df.rename(columns={
                    "日期": "datetime",
                    "收盘": "close",
                    "成交量": "volume",
                    "换手率": "turnover",
                })
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["instrument"] = inst
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
                records.append(df[["datetime", "instrument", "close", "volume", "turnover"]])
                success = True
                break
            except Exception as e:
                logger.warning(f"akshare fallback failed for {inst} (attempt {attempt + 1}/3): {e}")
                time.sleep(delay * 2)
        if not success:
            failed.append(inst)
        if idx % 100 == 0:
            logger.info(f"... fallback fetched {idx}/{len(instruments)} (success={len(records)}, failed={len(failed)})")
        time.sleep(delay)

    if not records:
        raise RuntimeError("No data fetched from akshare fallback.")
    logger.info(f"Fetch complete: {len(records)} stocks succeeded, {len(failed)} stocks failed.")
    df = pd.concat(records, ignore_index=True)
    df = df.set_index(["datetime", "instrument"]).sort_index()

    # Fetch stock name via akshare (industry not available in stock_info_a_code_name)
    stock_info = None
    try:
        import akshare as ak
        info = ak.stock_info_a_code_name()
        info["instrument"] = info["code"].apply(
            lambda c: f"SH{c}" if str(c).startswith("6") else f"SZ{c}"
        )
        stock_info = info.set_index("instrument")[["name"]]
    except Exception as e:
        logger.warning(f"Could not fetch stock info from akshare: {e}")

    return df, stock_info


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    # ---- date range --------------------------------------------------
    if args.trade_date:
        trade_date = pd.Timestamp(args.trade_date)
    else:
        trade_date = pd.Timestamp.now().normalize()
    start_date = (trade_date - pd.Timedelta(days=args.lookback + 10)).strftime("%Y-%m-%d")
    end_date = trade_date.strftime("%Y-%m-%d")
    logger.info(f"Trade date: {trade_date.date()}, range: {start_date} ~ {end_date}")

    # ---- instrument list ---------------------------------------------
    instruments = resolve_instruments(args.instruments)
    if isinstance(instruments, list):
        logger.info(f"Custom instrument list: {len(instruments)} stocks")
    else:
        logger.info(f"Instrument pool: {instruments}")

    # ---- fetch data --------------------------------------------------
    if args.fallback_tushare:
        # Tushare fallback
        if instruments == "all":
            logger.info("Tushare fallback: using 'all' (will fetch full market per trading day)")
        turnover_kwargs = {}
        if args.tushare_token:
            turnover_kwargs["api_token"] = args.tushare_token
        df, stock_info = fetch_fallback_tushare(instruments, start_date, end_date, **turnover_kwargs)
    elif args.fallback_akshare:
        # Need to expand pool to actual symbols when "all" is given
        if instruments == "all":
            logger.info("Resolving 'all' to full A-share list via akshare ...")
            import akshare as ak
            # stock_zh_a_spot_em may be blocked on some networks; use stock_info_a_code_name instead
            info = ak.stock_info_a_code_name()
            codes = info["code"].astype(str).tolist()
            instruments = [f"SH{c}" if c.startswith("6") else f"SZ{c}" for c in codes if c.isdigit()]
            logger.info(f"Total {len(instruments)} stocks in market")
        df, stock_info = fetch_fallback_akshare(instruments, start_date, end_date)
    else:
        turnover_kwargs = {}
        if args.turnover_source == "csv":
            if not args.turnover_csv:
                raise ValueError("--turnover-csv is required when --turnover-source=csv")
            turnover_kwargs["path"] = args.turnover_csv
        if args.tushare_token:
            turnover_kwargs["api_token"] = args.tushare_token

        ds = HybridDataSource(
            provider_uri=args.provider_uri,
            turnover_source=args.turnover_source,
            turnover_kwargs=turnover_kwargs,
        )
        df = ds.fetch(instruments, start_date, end_date)
        try:
            stock_info = ds.fetch_stock_info()
        except Exception as e:
            logger.warning(f"Could not fetch stock info: {e}")
            stock_info = None

    if df.empty:
        logger.error("No data fetched. Exiting.")
        sys.exit(1)

    logger.info(f"Loaded data shape: {df.shape}, dates: {df.index.get_level_values('datetime').nunique()}")

    # ---- run selector ------------------------------------------------
    # ---- optional macro check ----------------------------------------
    if args.macro_check:
        try:
            macro = MacroCheck(tushare_token=args.tushare_token)
            macro.print_report(trade_date.strftime("%Y-%m-%d"))
        except Exception as e:
            logger.warning(f"Macro check skipped: {e}")

    # ---- run selector ------------------------------------------------
    selector = EnhancedGoldenTriangleSelector(
        lookback_days=args.lookback,
        observation_window=args.obs_window,
        volume_lookback=args.volume_lookback,
        volume_multiplier=args.volume_multiplier,
        turnover_threshold=args.turnover_threshold,
        predict_ma_ratio=args.predict_ma_ratio,
        predict_vol_ratio=args.predict_vol_ratio,
        predict_turnover=args.predict_turnover,
        forming_ma_ratio=args.forming_ma_ratio,
        forming_vol_ratio=args.forming_vol_ratio,
        forming_turnover=args.forming_turnover,
    )

    name_map = stock_info["name"] if stock_info is not None else None
    industry_map = stock_info["industry"] if stock_info is not None else None

    # Fetch ST list for filtering
    st_set = None
    try:
        from qlib.contrib.golden_triangle.data_source import TurnoverFetcher
        st_set = set(TurnoverFetcher.fetch_st_list())
        logger.info(f"Fetched {len(st_set)} ST stocks for exclusion.")
    except Exception as e:
        logger.warning(f"Could not fetch ST list: {e}")

    result = selector.select(
        df,
        trade_date=trade_date,
        mode="all",
        industry_map=industry_map,
        name_map=name_map,
        st_set=st_set,
    )

    # ---- optional industry filter ------------------------------------
    if args.industry_filter and not result.empty:
        keywords = [k.strip() for k in args.industry_filter.split(",")]
        mask = result["industry"].astype(str).apply(lambda x: any(k in x for k in keywords))
        result = result[mask].copy()
        logger.info(f"After industry filter ({keywords}): {len(result)} stocks")

    # ---- output ------------------------------------------------------
    if result.empty:
        logger.warning("No stocks passed all filters.")
    else:
        # 打印三种状态各自的数量
        counts = result["signal_type"].value_counts().to_dict()
        logger.info(
            f"Final selection: {len(result)} stocks "
            f"(Confirmed={counts.get('confirmed', 0)}, "
            f"Predicting={counts.get('predicting', 0)}, "
            f"Forming={counts.get('forming', 0)})"
        )
        print("\n" + result.to_string(index=False))

    # Determine output path
    # 固定输出目录，文件名以年月日结尾
    out_dir = Path("examples/golden_triangle/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.output if args.output else "result"
    base_path = Path(base_name)
    stem = base_path.stem
    ext = base_path.suffix if base_path.suffix else ".csv"
    out_path = out_dir / f"{stem}_{trade_date.strftime('%Y%m%d')}{ext}"
    if out_path.suffix.lower() == ".xlsx":
        result.to_excel(out_path, index=False)
    else:
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Result saved to: {out_path.resolve()}")
    print(f"结果已保存: {out_path.resolve()}")

    # ---- copy to OpenClaw workspace for easy access ------------------
    try:
        import shutil
        workspace = Path.home() / ".openclaw" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        dest = workspace / out_path.name
        shutil.copy(str(out_path.resolve()), str(dest))
        logger.info(f"Copied to OpenClaw workspace: {dest}")
    except Exception as e:
        logger.debug(f"Could not copy to workspace: {e}")


if __name__ == "__main__":
    main()
