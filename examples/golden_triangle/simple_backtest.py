#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Golden Triangle (有效买托) Simple Backtest
============================================

A lightweight, self-contained backtest that does **not** require Qlib binary
data.  It combines the Golden Triangle selector with position management and
automatic trading, and outputs daily net-value + performance metrics.

Data sources:
- Default: fetch OHLCV + turnover from Tushare (bulk by trade_date, fastest).
- Alternative: fetch from AKShare per stock (slower).
- Or supply your own CSV.

Usage::

    # Tushare (recommended)
    python simple_backtest.py \
        --data-source tushare \
        --tushare-token your_token \
        --start 2024-01-01 \
        --end 2024-06-30 \
        --max-positions 10 \
        --holding-period 5 \
        --account 1000000

    # AKShare per stock (no token needed, slower)
    python simple_backtest.py \
        --data-source akshare \
        --start 2024-01-01 \
        --end 2024-03-31 \
        --max-positions 5

    # Custom CSV (columns: datetime,instrument,open,high,low,close,volume,turnover)
    python simple_backtest.py \
        --data-source csv \
        --csv-path data.csv \
        --start 2024-01-01 \
        --end 2024-12-31
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qlib.contrib.golden_triangle.enhanced_selector import EnhancedGoldenTriangleSelector


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_tushare(start_date: str, end_date: str, token: Optional[str], delay: float = 0.3) -> pd.DataFrame:
    try:
        import tushare as ts
    except ImportError:
        raise ImportError("tushare is required. pip install tushare")

    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("Tushare token required. Pass --tushare-token or set TUSHARE_TOKEN env var.")

    pro = ts.pro_api(token)
    cal = pro.trade_cal(
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        is_open="1",
    )
    if cal is None or cal.empty:
        raise RuntimeError("Tushare returned empty trade calendar.")
    trade_dates = sorted(cal["cal_date"].tolist())
    logger.info(f"Tushare: {len(trade_dates)} trading days in range")

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
        if idx % 100 == 0:
            logger.info(f"... tushare processed {idx}/{len(trade_dates)} days")
        time.sleep(delay)

    if not records:
        raise RuntimeError("No data fetched from tushare.")
    df = pd.concat(records, ignore_index=True)
    df = df.set_index(["datetime", "instrument"]).sort_index()
    return df


def fetch_akshare(start_date: str, end_date: str, delay: float = 0.5) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare is required. pip install akshare")

    info = ak.stock_info_a_code_name()
    codes = info["code"].astype(str).tolist()
    instruments = [f"SH{c}" if c.startswith("6") else f"SZ{c}" for c in codes if c.isdigit()]
    logger.info(f"AKShare: fetching {len(instruments)} stocks ...")

    records = []
    failed = []
    for idx, inst in enumerate(instruments, 1):
        symbol = inst[-6:]
        success = False
        for attempt in range(3):
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
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "换手率": "turnover",
                })
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["instrument"] = inst
                for col in ["open", "high", "low", "close", "volume", "turnover"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                records.append(df[["datetime", "instrument", "open", "high", "low", "close", "volume", "turnover"]])
                success = True
                break
            except Exception as e:
                logger.warning(f"akshare {inst} attempt {attempt + 1}/3 failed: {e}")
                time.sleep(delay * 2)
        if not success:
            failed.append(inst)
        if idx % 100 == 0:
            logger.info(f"... akshare fetched {idx}/{len(instruments)} (success={len(records)}, failed={len(failed)})")
        time.sleep(delay)

    if not records:
        raise RuntimeError("No data fetched from akshare.")
    df = pd.concat(records, ignore_index=True)
    df = df.set_index(["datetime", "instrument"]).sort_index()
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    dt_col = "datetime" if "datetime" in df.columns else "date"
    df[dt_col] = pd.to_datetime(df[dt_col])
    if "instrument" not in df.columns and "code" in df.columns:
        df["instrument"] = df["code"].apply(
            lambda c: f"SH{c}" if str(c).startswith("6") else f"SZ{c}"
        )
    df = df.set_index([dt_col, "instrument"]).sort_index()
    return df


# ---------------------------------------------------------------------------
# Portfolio & execution
# ---------------------------------------------------------------------------
@dataclass
class Holding:
    amount: float
    entry_price: float
    entry_date: pd.Timestamp
    days_held: int = 0


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Holding] = field(default_factory=dict)
    transactions: List[dict] = field(default_factory=list)

    def market_value(self, prices: pd.Series) -> float:
        stock_value = sum(
            h.amount * prices.get(inst, h.entry_price)
            for inst, h in self.positions.items()
        )
        return self.cash + stock_value

    def buy(self, date: pd.Timestamp, inst: str, price: float, amount: float, cost_rate: float):
        amount = int(amount)
        if amount <= 0:
            return
        value = amount * price
        cost = max(value * cost_rate, 5)
        total = value + cost
        if total > self.cash:
            # Adjust down to available cash.
            affordable = int((self.cash - cost) / price)
            if affordable <= 0:
                return
            amount = affordable
            value = amount * price
            cost = max(value * cost_rate, 5)
            total = value + cost
        self.cash -= total
        self.positions[inst] = Holding(amount=amount, entry_price=price, entry_date=date)
        self.transactions.append({
            "datetime": date,
            "instrument": inst,
            "action": "BUY",
            "price": price,
            "amount": amount,
            "cost": cost,
        })

    def sell(self, date: pd.Timestamp, inst: str, price: float, amount: float, cost_rate: float):
        holding = self.positions.get(inst)
        if holding is None or holding.amount <= 0:
            return
        amount = min(int(amount), holding.amount)
        value = amount * price
        cost = max(value * cost_rate, 5)
        self.cash += value - cost
        holding.amount -= amount
        if holding.amount <= 0:
            del self.positions[inst]
        self.transactions.append({
            "datetime": date,
            "instrument": inst,
            "action": "SELL",
            "price": price,
            "amount": amount,
            "cost": cost,
        })


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    selector: EnhancedGoldenTriangleSelector,
    max_positions: int,
    holding_period: int,
    sell_on_exit: bool,
    risk_degree: float,
    account: float,
    open_cost: float,
    close_cost: float,
    benchmark: Optional[pd.Series] = None,
    execution_price: str = "open",
) -> pd.DataFrame:
    """
    Vectorized-ish event backtest.

    On each trading day:
    1. Run selector with data up to previous close.
    2. Sell expired / exit-signal positions at today's ``execution_price``.
    3. Buy newly selected stocks at today's ``execution_price`` with equal weight.
    """
    dates = df.index.get_level_values("datetime").unique()
    dates = dates[(dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))]
    dates = sorted(dates)
    if len(dates) == 0:
        raise ValueError("No trading days in selected range.")

    portfolio = Portfolio(cash=account)
    nav_records = []

    all_instruments = df.index.get_level_values("instrument").unique().tolist()

    # Precompute selector signals once for the whole dataset to avoid repeated
    # groupby/rolling calculations on every trading day.
    logger.info("Precomputing selector signals ...")
    df = selector.precompute(df)

    for i, trade_date in enumerate(dates):
        # Prices for today.
        try:
            today = df.loc[trade_date]
        except KeyError:
            continue
        prices = today[execution_price]

        # ---- Update holding days ------------------------------------------------
        for inst, holding in portfolio.positions.items():
            holding.days_held += 1

        # ---- Run selector using data up to previous close -----------------------
        hist_end = trade_date - pd.Timedelta(days=1)
        hist_dates = dates[:i]
        hist_dates = [d for d in hist_dates if d <= hist_end]
        candidates_confirmed: List[str] = []
        candidates_predicting: List[str] = []
        if len(hist_dates) >= selector.min_listing_days:
            try:
                # df has been precomputed; select() will only slice and filter.
                # 一次性取出三状态结果，再按 signal_type 拆分。
                result = selector.select(df, trade_date=hist_end, mode="all")
                if not result.empty:
                    candidates_confirmed = result.loc[
                        result["signal_type"] == "confirmed", "instrument"
                    ].tolist()
                    candidates_predicting = result.loc[
                        result["signal_type"] == "predicting", "instrument"
                    ].tolist()
            except Exception as e:
                logger.debug(f"Selector failed on {hist_end.date()}: {e}")

        # ---- Sell ----------------------------------------------------------------
        # 卖出逻辑：若开启 sell_on_exit，当票不在 confirmed 也不在 predicting 中时卖出
        # Forming 仅观察，不持有。
        selected_buy_insts = set(candidates_confirmed) | set(candidates_predicting)
        for inst in list(portfolio.positions.keys()):
            holding = portfolio.positions[inst]
            price = prices.get(inst)
            if price is None or pd.isna(price) or price <= 0:
                continue
            expired = holding.days_held >= holding_period
            exit_signal = sell_on_exit and inst not in selected_buy_insts
            if expired or exit_signal:
                portfolio.sell(trade_date, inst, price, holding.amount, close_cost)

        # ---- Buy -----------------------------------------------------------------
        # 买入逻辑分层：
        # 1. Confirmed 优先占满仓位，每只等权分全额资金；
        # 2. Predicting 填补剩余仓位，每只等权分 50% 资金（半仓试探）；
        # 3. Forming 不买入。
        current_insts = set(portfolio.positions.keys())
        new_confirmed = [inst for inst in candidates_confirmed if inst not in current_insts]
        new_predicting = [inst for inst in candidates_predicting if inst not in current_insts]

        available_slots = max_positions - len(current_insts)
        new_confirmed = new_confirmed[:available_slots]
        available_slots -= len(new_confirmed)
        new_predicting = new_predicting[:available_slots]

        n_confirmed = len(new_confirmed)
        n_predicting = len(new_predicting)

        if n_confirmed > 0 or n_predicting > 0:
            total_value = portfolio.market_value(prices)
            investable = total_value * risk_degree
            reserved_stock_value = sum(
                h.amount * prices.get(inst, h.entry_price)
                for inst, h in portfolio.positions.items()
            )
            budget = min(portfolio.cash, investable - reserved_stock_value)

            # 以“全仓等权”为 1 个 slot；Predicting 仅占 0.5 个 slot。
            total_slots = n_confirmed + 0.5 * n_predicting
            if total_slots > 0:
                budget_per_slot = budget / total_slots

                # Confirmed：全额买入
                for inst in new_confirmed:
                    price = prices.get(inst)
                    if price is None or pd.isna(price) or price <= 0:
                        continue
                    amount = budget_per_slot / price
                    portfolio.buy(trade_date, inst, price, amount, open_cost)

                # Predicting：半仓试探买入
                for inst in new_predicting:
                    price = prices.get(inst)
                    if price is None or pd.isna(price) or price <= 0:
                        continue
                    amount = (0.5 * budget_per_slot) / price
                    portfolio.buy(trade_date, inst, price, amount, open_cost)

        # ---- Record NAV ----------------------------------------------------------
        nav = portfolio.market_value(prices)
        nav_records.append({
            "datetime": trade_date,
            "nav": nav,
            "cash": portfolio.cash,
            "positions": len(portfolio.positions),
        })

    nav_df = pd.DataFrame(nav_records).set_index("datetime")
    nav_df["daily_return"] = nav_df["nav"].pct_change()
    nav_df["cum_return"] = nav_df["nav"] / account - 1
    if benchmark is not None:
        bench_aligned = benchmark.reindex(nav_df.index).ffill()
        nav_df["bench"] = bench_aligned / bench_aligned.iloc[0] * account
        nav_df["bench_cum_return"] = nav_df["bench"] / account - 1
    return nav_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calc_metrics(nav_df: pd.DataFrame, risk_free: float = 0.0) -> pd.DataFrame:
    returns = nav_df["daily_return"].dropna()
    if returns.empty or returns.std() == 0:
        return pd.DataFrame()

    ann_factor = 252
    ann_return = returns.mean() * ann_factor
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else np.nan

    cum = nav_df["nav"] / nav_df["nav"].iloc[0]
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    total_return = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1
    win_rate = (returns > 0).mean()

    metrics = {
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "start_nav": nav_df["nav"].iloc[0],
        "end_nav": nav_df["nav"].iloc[-1],
    }
    return pd.DataFrame([metrics])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Golden Triangle Simple Backtest")
    p.add_argument("--data-source", default="tushare", choices=["tushare", "akshare", "csv"], help="Data source")
    p.add_argument("--csv-path", default=None, help="CSV path when data-source=csv")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--tushare-token", default=None, help="Tushare token (or set TUSHARE_TOKEN env var)")
    p.add_argument("--tushare-delay", type=float, default=0.3, help="Tushare API sleep seconds")
    p.add_argument("--akshare-delay", type=float, default=0.5, help="AKShare per-stock sleep seconds")
    p.add_argument("--account", type=float, default=1_000_000, help="Initial cash")
    p.add_argument("--max-positions", type=int, default=10, help="Max simultaneous holdings")
    p.add_argument("--holding-period", type=int, default=5, help="Holding period in trading days")
    p.add_argument("--sell-on-exit-signal", action="store_true", help="Sell when stock exits selection")
    p.add_argument("--lookback", type=int, default=35, help="Selector lookback days")
    p.add_argument("--obs-window", type=int, default=3, help="Golden triangle observation window")
    p.add_argument("--volume-lookback", type=int, default=5, help="Volume average lookback")
    p.add_argument("--volume-multiplier", type=float, default=1.5, help="Volume surge threshold")
    p.add_argument("--turnover-threshold", type=float, default=3.0, help="Min turnover %%")
    p.add_argument("--predict-ma-ratio", type=float, default=0.985, help="Predicting state min MA10/MA20 ratio")
    p.add_argument("--predict-vol-ratio", type=float, default=1.3, help="Predicting state min volume ratio")
    p.add_argument("--predict-turnover", type=float, default=2.5, help="Predicting state min turnover %%")
    p.add_argument("--forming-ma-ratio", type=float, default=0.97, help="Forming state min MA10/MA20 ratio")
    p.add_argument("--forming-vol-ratio", type=float, default=1.1, help="Forming state min volume ratio")
    p.add_argument("--forming-turnover", type=float, default=2.0, help="Forming state min turnover %%")
    p.add_argument("--open-cost", type=float, default=0.0005, help="Open commission")
    p.add_argument("--close-cost", type=float, default=0.0015, help="Close commission")
    p.add_argument("--risk-degree", type=float, default=0.95, help="Max fraction of total value in stocks")
    p.add_argument("--execution-price", default="open", choices=["open", "close"], help="Trade execution price")
    p.add_argument("--benchmark", default="SH000300", help="Benchmark code (Tushare only)")
    p.add_argument("--output-dir", default="examples/golden_triangle/simple_backtest_output", help="Output dir")
    p.add_argument("--no-plots", action="store_true", help="Skip plotting")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    # ---- Load data -----------------------------------------------------------
    logger.info(f"Loading data from {args.data_source} ...")
    if args.data_source == "tushare":
        df = fetch_tushare(args.start, args.end, args.tushare_token, args.tushare_delay)
    elif args.data_source == "akshare":
        df = fetch_akshare(args.start, args.end, args.akshare_delay)
    elif args.data_source == "csv":
        if not args.csv_path:
            raise ValueError("--csv-path required when data-source=csv")
        df = load_csv(args.csv_path)
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    if df.empty:
        raise RuntimeError("Empty dataset.")
    logger.info(f"Data loaded: {len(df)} rows, {df.index.get_level_values('instrument').nunique()} instruments")

    # ---- Optional benchmark --------------------------------------------------
    benchmark = None
    if args.data_source == "tushare" and args.benchmark:
        try:
            import tushare as ts
            token = args.tushare_token or os.environ.get("TUSHARE_TOKEN")
            pro = ts.pro_api(token)
            ts_code = f"{args.benchmark[-6:]}.{args.benchmark[:2]}"
            bench_df = pro.index_daily(
                ts_code=ts_code,
                start_date=args.start.replace("-", ""),
                end_date=args.end.replace("-", ""),
            )
            if bench_df is not None and not bench_df.empty:
                bench_df = bench_df.rename(columns={"trade_date": "datetime", "close": "close"})
                bench_df["datetime"] = pd.to_datetime(bench_df["datetime"])
                benchmark = bench_df.set_index("datetime")["close"].sort_index()
        except Exception as e:
            logger.warning(f"Could not fetch benchmark: {e}")

    # ---- Run backtest --------------------------------------------------------
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

    logger.info("Running backtest ...")
    nav_df = run_backtest(
        df=df,
        start_date=args.start,
        end_date=args.end,
        selector=selector,
        max_positions=args.max_positions,
        holding_period=args.holding_period,
        sell_on_exit=args.sell_on_exit_signal,
        risk_degree=args.risk_degree,
        account=args.account,
        open_cost=args.open_cost,
        close_cost=args.close_cost,
        benchmark=benchmark,
        execution_price=args.execution_price,
    )

    # ---- Save results --------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nav_path = out_dir / "nav.csv"
    nav_df.to_csv(nav_path, encoding="utf-8-sig")
    logger.info(f"NAV saved: {nav_path}")

    metrics = calc_metrics(nav_df)
    if not metrics.empty:
        metrics_path = out_dir / "metrics.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        logger.info(f"Metrics saved: {metrics_path}")

    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(nav_df.tail(10).to_string())
    print("\n绩效指标:")
    print(metrics.to_string(index=False))

    # ---- Plot ----------------------------------------------------------------
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(nav_df.index, nav_df["nav"], label="Strategy", linewidth=1.5)
            if "bench" in nav_df.columns:
                ax.plot(nav_df.index, nav_df["bench"], label="Benchmark", linewidth=1.5)
            ax.set_title("Golden Triangle Simple Backtest")
            ax.set_xlabel("Date")
            ax.set_ylabel("Net Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_path = out_dir / "nav_curve.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Plot saved: {plot_path}")
        except Exception as e:
            logger.warning(f"Could not plot: {e}")

    print(f"\n结果保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
