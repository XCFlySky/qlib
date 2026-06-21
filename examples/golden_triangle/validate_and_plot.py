#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Golden Triangle 选股结果验证与可视化

用法::

    python validate_and_plot.py --input result_20260612.csv --output-dir plots

功能:
1. 读取选股结果 CSV，逐只获取历史 K 线
2. 重新计算 MA5/MA10/MA20，检测金叉点
3. 验证 cross_date 当天是否满足"两个金叉均已发生 + 多头排列"
4. 为每只股票生成一张 K 线+均线+金叉标记图
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# --------------------------------------------------------------------
# 数据获取
# --------------------------------------------------------------------
def fetch_stock_data_tushare(ts_code: str, start_date: str, end_date: str, api_token: str):
    """通过 tushare pro 获取单只股票历史 K 线。"""
    try:
        import tushare as ts
    except ImportError:
        raise ImportError("tushare is required. pip install tushare")

    pro = ts.pro_api(api_token)
    df = pro.daily(
        ts_code=ts_code,
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )
    if df is None or df.empty:
        return None

    df = df.rename(
        columns={
            "trade_date": "Date",
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "vol": "Volume",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce") * 100  # 手 -> 股
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df


def fetch_stock_data_akshare(symbol: str, start_date: str, end_date: str, delay: float = 1.2):
    """通过 akshare 获取单只股票历史 K 线（前复权），带重试与延迟。"""
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare is required. pip install akshare")

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
                return None

            df = df.rename(
                columns={
                    "日期": "Date",
                    "开盘": "Open",
                    "收盘": "Close",
                    "最高": "High",
                    "最低": "Low",
                    "成交量": "Volume",
                }
            )
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            return df
        except Exception as e:
            logger.warning(f"  akshare fetch failed for {symbol} (attempt {attempt + 1}/3): {e}")
            time.sleep(delay * (attempt + 1))
    return None


def inst_to_ts_code(inst: str) -> str:
    """SH600000 -> 600000.SH"""
    return f"{inst[-6:]}.{inst[:2]}"


def fetch_stock_data(inst: str, start_date: str, end_date: str, api_token: str = None):
    """优先用 tushare，失败回退到 akshare。"""
    if api_token:
        try:
            ts_code = inst_to_ts_code(inst)
            return fetch_stock_data_tushare(ts_code, start_date, end_date, api_token)
        except Exception as e:
            logger.warning(f"  tushare failed for {inst}, fallback to akshare: {e}")
    return fetch_stock_data_akshare(inst[-6:], start_date, end_date)


# --------------------------------------------------------------------
# 指标与验证
# --------------------------------------------------------------------
def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    df["MA5"] = df["Close"].rolling(window=5, min_periods=5).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=10).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    return df


def find_cross_dates(df: pd.DataFrame, fast: str, slow: str):
    """返回 fast 上穿 slow 的所有日期列表。"""
    mask = (df[fast] > df[slow]) & (df[fast].shift(1) <= df[slow].shift(1))
    return df.index[mask].tolist()


def validate_and_plot(result_csv: str, output_dir: str, days_before: int = 30, days_after: int = 5, api_token: str = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = pd.read_csv(result_csv, encoding="utf-8-sig")
    if result.empty:
        logger.warning("选股结果为空，没什么可验证的。")
        return

    # 统一列名（兼容中英文双语表头）
    col_map = {}
    for c in result.columns:
        cl = c.lower().replace("（", "(").replace("）", ")")
        if "instrument" in cl:
            col_map[c] = "instrument"
        elif "cross_date" in cl:
            col_map[c] = "cross_date"
        elif "ma5" in cl and "ma10" not in cl and "ma20" not in cl:
            col_map[c] = "ma5"
        elif "ma10" in cl:
            col_map[c] = "ma10"
        elif "ma20" in cl:
            col_map[c] = "ma20"
    result = result.rename(columns=col_map)

    report_rows = []

    for idx, row in result.iterrows():
        inst = str(row["instrument"]).strip()
        cross_date_raw = str(row["cross_date"]).split()[0]
        symbol = inst[-6:]  # SH600000 -> 600000

        logger.info(f"[{idx + 1}/{len(result)}] 验证 {inst} …")

        try:
            cross_date = pd.Timestamp(cross_date_raw)
            # 多取 25 天用于计算 MA20
            fetch_start = (cross_date - pd.Timedelta(days=days_before + 25)).strftime("%Y-%m-%d")
            fetch_end = (cross_date + pd.Timedelta(days=days_after)).strftime("%Y-%m-%d")

            df = fetch_stock_data(inst, fetch_start, fetch_end, api_token)
            if df is None or df.empty:
                logger.warning(f"  {inst} 获取不到数据，跳过")
                continue

            df = add_mas(df)

            # 检测金叉
            cross_5_10_dates = find_cross_dates(df, "MA5", "MA10")
            cross_10_20_dates = find_cross_dates(df, "MA10", "MA20")

            # 验证 cross_date 当天是否多头排列
            if cross_date in df.index:
                ma5_cd = df.loc[cross_date, "MA5"]
                ma10_cd = df.loc[cross_date, "MA10"]
                ma20_cd = df.loc[cross_date, "MA20"]
                bull_ok = (ma5_cd > ma10_cd) and (ma10_cd > ma20_cd)
            else:
                bull_ok = False
                ma5_cd = ma10_cd = ma20_cd = np.nan

            # 在 cross_date 前后 7 天作为观察窗口，检查两个金叉是否都已发生
            obs_start = cross_date - pd.Timedelta(days=7)
            obs_end = cross_date + pd.Timedelta(days=7)
            c5_obs = [d for d in cross_5_10_dates if obs_start <= d <= obs_end]
            c10_obs = [d for d in cross_10_20_dates if obs_start <= d <= obs_end]
            both_ok = len(c5_obs) > 0 and len(c10_obs) > 0

            status = "PASS" if both_ok and bull_ok else "FAIL"
            report_rows.append(
                {
                    "instrument": inst,
                    "cross_date": cross_date_raw,
                    "cross_5_10_nearby": ";".join([d.strftime("%Y-%m-%d") for d in c5_obs]),
                    "cross_10_20_nearby": ";".join([d.strftime("%Y-%m-%d") for d in c10_obs]),
                    "bull_arrange_on_cross": bull_ok,
                    "both_crosses_nearby": both_ok,
                    "status": status,
                }
            )

            # ---------- 画图 ----------
            plot_start = cross_date - pd.Timedelta(days=days_before)
            plot_df = df.loc[plot_start:].copy()

            addplots = [
                mpf.make_addplot(plot_df["MA5"], color="orange", width=0.7, label="MA5"),
                mpf.make_addplot(plot_df["MA10"], color="blue", width=0.7, label="MA10"),
                mpf.make_addplot(plot_df["MA20"], color="purple", width=0.7, label="MA20"),
            ]

            # 在金叉日画三角/倒三角标记
            plot_df["mark_5_10"] = np.where(
                plot_df.index.isin(cross_5_10_dates), plot_df["Close"] * 1.03, np.nan
            )
            plot_df["mark_10_20"] = np.where(
                plot_df.index.isin(cross_10_20_dates), plot_df["Close"] * 0.97, np.nan
            )
            addplots.append(
                mpf.make_addplot(
                    plot_df["mark_5_10"], type="scatter", markersize=80, marker="^", color="lime"
                )
            )
            addplots.append(
                mpf.make_addplot(
                    plot_df["mark_10_20"], type="scatter", markersize=80, marker="v", color="magenta"
                )
            )

            vlines = dict(vlines=cross_date, colors="red", linewidths=1.2, alpha=0.8)

            title = f"{inst}  cross={cross_date_raw}  status={status}"
            save_path = output_dir / f"{inst}_{cross_date_raw}.png"

            mpf.plot(
                plot_df,
                type="candle",
                style="charles",
                title=title,
                ylabel="Price",
                volume=True,
                addplot=addplots,
                vlines=vlines,
                figsize=(14, 8),
                savefig=dict(fname=save_path, dpi=150, bbox_inches="tight"),
            )
            plt.close("all")
            logger.info(f"  图表已保存: {save_path}")

        except Exception as e:
            logger.error(f"  {inst} 处理异常: {e}")

    # ---------- 打印报告 ----------
    print("\n" + "=" * 90)
    print("Golden Triangle 验证报告")
    print("=" * 90)
    pass_count = sum(1 for r in report_rows if r["status"] == "PASS")
    print(f"总计: {len(report_rows)} 只 | 通过: {pass_count} | 失败: {len(report_rows) - pass_count}\n")

    for r in report_rows:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(
            f"{icon} {r['instrument']:12s}  cross={r['cross_date']}  "
            f"MA5↑MA10={r['cross_5_10_nearby'] or 'None':20s}  "
            f"MA10↑MA20={r['cross_10_20_nearby'] or 'None':20s}  "
            f"bull={r['bull_arrange_on_cross']}"
        )

    report_path = output_dir / "validation_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n详细报告已保存: {report_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Golden Triangle 选股结果验证与可视化")
    p.add_argument("--input", default="examples/golden_triangle/result_all.csv", help="选股结果 CSV")
    p.add_argument("--output-dir", default="examples/golden_triangle/plots", help="图表输出目录")
    p.add_argument("--days-before", type=int, default=30, help="cross_date 前展示多少天")
    p.add_argument("--days-after", type=int, default=5, help="cross_date 后展示多少天")
    p.add_argument("--tushare-token", default=None, help="Tushare pro token（优先使用 tushare 拉数据，更稳定）")
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    api_token = args.tushare_token or os.environ.get("TUSHARE_TOKEN")
    validate_and_plot(args.input, args.output_dir, args.days_before, args.days_after, api_token)
