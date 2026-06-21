#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Golden Triangle (有效买托) Demo – 零依赖即可运行
=====================================================

本脚本使用**模拟数据**演示选股核心逻辑，不依赖网络、不依赖 Qlib 本地数据。
适合：
1. 快速理解策略逻辑
2. 验证代码安装是否正确
3. 在无网络环境下跑通全流程

用法：
    python demo.py
"""

import sys
from pathlib import Path

# 确保能 import 到 qlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from qlib.contrib.golden_triangle.selector import GoldenTriangleSelector


def build_demo_data(n_stocks: int = 5, n_days: int = 35):
    """
    构造模拟行情数据。
    其中只有 2 只股票会出现有效买托信号。
    """
    np.random.seed(42)
    dates = pd.date_range("2024-08-01", periods=n_days, freq="B")
    records = []

    for i in range(n_stocks):
        inst = f"SH{600000 + i * 10}"
        close = 10.0 + np.cumsum(np.random.randn(n_days) * 0.3)
        volume = np.random.randint(5000, 15000, size=n_days)
        turnover = np.random.uniform(1.0, 2.5, size=n_days)

        # i=2 -> SH600020: 出现完整金叉三角 + 量能 + 换手率，应被选中
        if i == 2:
            close[:] = 10.0
            close[-3] = 11.0   # 触发金叉
            close[-2] = 15.0
            close[-1] = 20.0
            volume[-3] = 50000  # 量比 >= 1.5x
            turnover[-3] = 5.5  # 换手率 > 3%

        # i=4 -> SH600040: 也有金叉和量能，但换手率不足，应被过滤
        if i == 4:
            close[:] = 10.0
            close[-3] = 11.0
            close[-2] = 15.0
            close[-1] = 20.0
            volume[-3] = 50000
            turnover[-3] = 2.0  # 不足 3%，应被过滤

        for d, c, v, t in zip(dates, close, volume, turnover):
            records.append(
                {
                    "datetime": d,
                    "instrument": inst,
                    "close": float(c),
                    "volume": int(v),
                    "turnover": float(t),
                }
            )

    df = pd.DataFrame(records)
    df = df.set_index(["datetime", "instrument"])
    return df


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("=" * 60)
    print("有效买托策略 - 模拟数据演示")
    print("=" * 60)

    df = build_demo_data(n_stocks=5, n_days=35)
    print(f"\n[数据] 模拟数据: {df.index.get_level_values('instrument').nunique()} 只股票, "
          f"{df.index.get_level_values('datetime').nunique()} 个交易日\n")

    selector = GoldenTriangleSelector(
        lookback_days=35,
        observation_window=3,
        volume_multiplier=1.5,
        turnover_threshold=3.0,
    )

    trade_date = df.index.get_level_values("datetime").max()
    result = selector.select(df, trade_date=trade_date)

    print("\n[结果] 选股结果:\n")
    if result.empty:
        print("  (无股票满足所有条件)")
    else:
        # 只展示关键列
        show_cols = [
            "instrument", "cross_date", "ma5", "ma10", "ma20",
            "volume", "avg_volume_5", "volume_ratio", "turnover"
        ]
        print(result[[c for c in show_cols if c in result.columns]].to_string(index=False))

    print(f"\n[完成] 共选出 {len(result)} 只股票（预期: 1 只 SH600020，SH600040 因换手率不足被过滤）")
    print("=" * 60)

    # 保存到本地 CSV，方便查看
    out = Path(__file__).parent / "result_all.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 结果已保存: {out}")


if __name__ == "__main__":
    main()
