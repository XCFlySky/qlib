#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Golden Triangle 参数优化器
==========================
网格搜索 + 样本外验证 + 稳健性评分
"""

import argparse
import itertools
import json
import multiprocessing
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# 从现有项目导入
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qlib.contrib.golden_triangle.enhanced_selector import EnhancedGoldenTriangleSelector

# ---------------------------------------------------------------------------
# 时间区间划分（硬性要求，优化过程中不可修改）
# ---------------------------------------------------------------------------
TRAIN_START, TRAIN_END = "2020-01-01", "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2023-12-31"
TEST_START, TEST_END = "2024-01-01", "2024-06-30"

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSet:
    lookback: int
    obs_window: int
    vol_lb: int
    vol_mul: float
    to_thresh: float
    holding_period: int


@dataclass
class BacktestResult:
    """单个参数组合在某一数据集上的回测结果"""
    param: ParamSet
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    annual_volatility: float
    win_rate: float
    trade_count: int
    robustness: float = 0.0  # 仅在训练集结果上填充
    val_sharpe: float = 0.0
    val_return: float = 0.0
    val_maxdd: float = 0.0
    eliminated_reason: str = ""


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
            "datetime": date, "instrument": inst, "action": "BUY",
            "price": price, "amount": amount, "cost": cost,
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
            "datetime": date, "instrument": inst, "action": "SELL",
            "price": price, "amount": amount, "cost": cost,
        })


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data(data_path: str) -> pd.DataFrame:
    """加载本地 CSV，要求列：datetime/date, instrument/code, open, high, low, close, volume, turnover"""
    df = pd.read_csv(data_path)
    df.columns = [c.lower().strip() for c in df.columns]

    dt_col = "datetime" if "datetime" in df.columns else "date"
    df[dt_col] = pd.to_datetime(df[dt_col])

    if "instrument" not in df.columns and "code" in df.columns:
        df["instrument"] = df["code"].apply(
            lambda c: f"SH{c}" if str(c).startswith("6") else f"SZ{c}"
        )

    df = df.set_index([dt_col, "instrument"]).sort_index()
    logger.info(
        f"Data loaded: {len(df)} rows, "
        f"{df.index.get_level_values('instrument').nunique()} instruments, "
        f"dates {df.index.get_level_values('datetime').min().date()} ~ "
        f"{df.index.get_level_values('datetime').max().date()}"
    )
    return df


def slice_with_buffer(df: pd.DataFrame, start: str, end: str, buffer_days: int = 60) -> pd.DataFrame:
    """按区间切分数据，并保留 start 前 buffer_days 天用于计算 MA20 / 均量"""
    start_dt = pd.Timestamp(start) - pd.Timedelta(days=buffer_days)
    end_dt = pd.Timestamp(end)
    mask = (df.index.get_level_values("datetime") >= start_dt) & (
        df.index.get_level_values("datetime") <= end_dt
    )
    return df[mask].copy()


# ---------------------------------------------------------------------------
# 回测引擎（已预计算 df 版本，避免网格搜索时重复 groupby）
# ---------------------------------------------------------------------------
def run_backtest_engine(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    selector: EnhancedGoldenTriangleSelector,
    max_positions: int = 10,
    holding_period: int = 5,
    sell_on_exit: bool = True,
    risk_degree: float = 0.95,
    account: float = 1_000_000,
    open_cost: float = 0.0005,
    close_cost: float = 0.0015,
    execution_price: str = "open",
) -> Tuple[pd.DataFrame, int]:
    """
    事件驱动回测引擎。
    要求 df 已经过 selector.precompute() 预处理，包含所有信号列。
    """
    dates = df.index.get_level_values("datetime").unique()
    dates = dates[(dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))]
    dates = sorted(dates)
    if len(dates) == 0:
        raise ValueError(f"No trading days between {start_date} and {end_date}.")

    portfolio = Portfolio(cash=account)
    nav_records = []

    for i, trade_date in enumerate(dates):
        try:
            today = df.loc[trade_date]
        except KeyError:
            continue
        prices = today[execution_price]

        # 更新持仓天数
        for holding in portfolio.positions.values():
            holding.days_held += 1

        # ---- 选股：用截至前一日的数据 ----
        hist_end = trade_date - pd.Timedelta(days=1)
        hist_dates = dates[:i]
        hist_dates = [d for d in hist_dates if d <= hist_end]
        candidates_confirmed: List[str] = []
        candidates_predicting: List[str] = []

        if len(hist_dates) >= selector.min_listing_days:
            try:
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

        # ---- 卖出：不在 confirmed / predicting 中的持仓卖出 ----
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

        # ---- 买入：Confirmed 全额，Predicting 半仓，Forming 不买 ----
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

            total_slots = n_confirmed + 0.5 * n_predicting
            if total_slots > 0:
                budget_per_slot = budget / total_slots

                for inst in new_confirmed:
                    price = prices.get(inst)
                    if price is None or pd.isna(price) or price <= 0:
                        continue
                    amount = budget_per_slot / price
                    portfolio.buy(trade_date, inst, price, amount, open_cost)

                for inst in new_predicting:
                    price = prices.get(inst)
                    if price is None or pd.isna(price) or price <= 0:
                        continue
                    amount = (0.5 * budget_per_slot) / price
                    portfolio.buy(trade_date, inst, price, amount, open_cost)

        # ---- 记录净值 ----
        nav = portfolio.market_value(prices)
        nav_records.append({
            "datetime": trade_date,
            "nav": nav,
            "cash": portfolio.cash,
            "positions": len(portfolio.positions),
        })

    nav_df = pd.DataFrame(nav_records).set_index("datetime")
    nav_df["daily_return"] = nav_df["nav"].pct_change()
    trade_count = sum(1 for t in portfolio.transactions if t["action"] == "BUY")
    return nav_df, trade_count


# ---------------------------------------------------------------------------
# 绩效指标
# ---------------------------------------------------------------------------
def calc_metrics(nav_df: pd.DataFrame, risk_free: float = 0.0) -> Dict:
    """计算年化收益、夏普、最大回撤、波动率、胜率"""
    returns = nav_df["daily_return"].dropna()
    if returns.empty or returns.std() == 0:
        return {
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "annual_volatility": 0.0,
            "win_rate": 0.0,
        }

    ann_factor = 252
    ann_return = returns.mean() * ann_factor
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0.0

    cum = nav_df["nav"] / nav_df["nav"].iloc[0]
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    win_rate = (returns > 0).mean()

    return {
        "annual_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "annual_volatility": ann_vol,
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# 单次回测包装
# ---------------------------------------------------------------------------
def run_single_backtest(
    param: ParamSet, df_precomputed: pd.DataFrame, start: str, end: str, account: float
) -> BacktestResult:
    """对单个参数组合在预计算 df 上跑回测，返回 BacktestResult"""
    selector = EnhancedGoldenTriangleSelector(
        lookback_days=param.lookback,
        observation_window=param.obs_window,
        volume_lookback=param.vol_lb,
        volume_multiplier=param.vol_mul,
        turnover_threshold=param.to_thresh,
    )

    try:
        nav_df, trade_count = run_backtest_engine(
            df_precomputed,
            start_date=start,
            end_date=end,
            selector=selector,
            holding_period=param.holding_period,
            account=account,
        )
        metrics = calc_metrics(nav_df)
        return BacktestResult(
            param=param,
            annual_return=metrics["annual_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            annual_volatility=metrics["annual_volatility"],
            win_rate=metrics["win_rate"],
            trade_count=trade_count,
        )
    except Exception as e:
        logger.error(f"Backtest failed for {param}: {e}")
        return BacktestResult(
            param=param,
            annual_return=0.0,
            sharpe_ratio=-999.0,
            max_drawdown=0.0,
            annual_volatility=0.0,
            win_rate=0.0,
            trade_count=0,
        )


# ---------------------------------------------------------------------------
# 并行运行 + 进度打印
# ---------------------------------------------------------------------------
def _run_single_backtest_tuple(args):
    """multiprocessing 需要可 pickle 的入口"""
    param, df_precomputed, start, end, account = args
    return run_single_backtest(param, df_precomputed, start, end, account)


def parallel_run(func, args_list: List, n_jobs: int) -> List:
    """使用 multiprocessing.Pool 并行，每完成 10% 打印进度与 ETA"""
    n_jobs = min(n_jobs, multiprocessing.cpu_count())
    total = len(args_list)
    if total == 0:
        return []

    results = []
    start_time = time.time()
    last_report = 0.0

    logger.info(f"Starting parallel run with {n_jobs} workers, total tasks={total}")
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(func, args_list), 1):
            results.append(res)
            progress = i / total
            if progress - last_report >= 0.1 or i == total:
                elapsed = time.time() - start_time
                eta = elapsed / i * (total - i) if i < total else 0
                logger.info(
                    f"Progress: {i}/{total} ({progress * 100:.0f}%), "
                    f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s"
                )
                last_report = progress
    return results


# ---------------------------------------------------------------------------
# 稳健性测试
# ---------------------------------------------------------------------------
def robustness_test(
    top_results: List[BacktestResult],
    df_train_precomputed: pd.DataFrame,
    account: float,
    n_jobs: int,
    perturb_pct: float = 0.1,
    min_robustness: float = 0.7,
) -> List[BacktestResult]:
    """
    对候选参数进行 ±10% 扰动测试。
    稳健分 = 1 - mean(|Sharpe_perturbed - Sharpe_base| / |Sharpe_base|)
    只保留稳健分 > min_robustness 的组合。
    """
    robust_results = []
    for base_res in top_results:
        param = base_res.param
        base_sharpe = base_res.sharpe_ratio

        perturbations = [
            ParamSet(param.lookback, param.obs_window, param.vol_lb,
                     param.vol_mul * (1 + perturb_pct), param.to_thresh, param.holding_period),
            ParamSet(param.lookback, param.obs_window, param.vol_lb,
                     param.vol_mul * (1 - perturb_pct), param.to_thresh, param.holding_period),
            ParamSet(param.lookback, param.obs_window, param.vol_lb,
                     param.vol_mul, param.to_thresh * (1 + perturb_pct), param.holding_period),
            ParamSet(param.lookback, param.obs_window, param.vol_lb,
                     param.vol_mul, param.to_thresh * (1 - perturb_pct), param.holding_period),
        ]

        perturbed_args = [
            (p, df_train_precomputed, TRAIN_START, TRAIN_END, account)
            for p in perturbations
        ]
        perturbed_results = parallel_run(_run_single_backtest_tuple, perturbed_args, n_jobs)

        rel_changes = []
        for pr in perturbed_results:
            ps = pr.sharpe_ratio
            if abs(base_sharpe) < 1e-6:
                rel_changes.append(1.0)
            else:
                rel_changes.append(abs(ps - base_sharpe) / abs(base_sharpe))

        robustness = 1 - float(np.mean(rel_changes))
        base_res.robustness = robustness

        if robustness >= min_robustness:
            robust_results.append(base_res)
            logger.info(
                f"Robustness passed: {param}, base_sharpe={base_sharpe:.3f}, "
                f"robustness={robustness:.3f}"
            )
        else:
            logger.info(
                f"Robustness failed: {param}, base_sharpe={base_sharpe:.3f}, "
                f"robustness={robustness:.3f}"
            )

    return robust_results


# ---------------------------------------------------------------------------
# 验证集盲测
# ---------------------------------------------------------------------------
def validation_test(
    robust_results: List[BacktestResult],
    df_val_precomputed: pd.DataFrame,
    account: float,
    n_jobs: int,
) -> List[BacktestResult]:
    """
    将训练集筛选出的稳健组合拿到验证集跑回测，应用淘汰规则：
    - 验证集夏普 < 训练集夏普的 50% → 淘汰
    - 验证集最大回撤 > 训练集回撤的 2 倍 → 淘汰
    - 验证集胜率 < 40% → 淘汰
    """
    val_args = [
        (r.param, df_val_precomputed, VAL_START, VAL_END, account)
        for r in robust_results
    ]
    val_results = parallel_run(_run_single_backtest_tuple, val_args, n_jobs)

    final_results = []
    for r, vr in zip(robust_results, val_results):
        r.val_sharpe = vr.sharpe_ratio
        r.val_return = vr.annual_return
        r.val_maxdd = vr.max_drawdown

        train_sharpe = r.sharpe_ratio
        train_maxdd = r.max_drawdown
        val_sharpe = r.val_sharpe
        val_maxdd = r.val_maxdd
        val_winrate = vr.win_rate

        if val_sharpe < 0.5 * train_sharpe:
            r.eliminated_reason = "val_sharpe < 0.5 * train_sharpe"
        elif val_maxdd < 2 * train_maxdd:  # 两者均为负数，< 表示回撤更深
            r.eliminated_reason = "val_maxdd > 2 * train_maxdd"
        elif val_winrate < 0.4:
            r.eliminated_reason = "val_winrate < 0.4"
        else:
            r.eliminated_reason = ""
            final_results.append(r)
            logger.info(
                f"Validation passed: {r.param}, "
                f"train_sharpe={train_sharpe:.3f}, val_sharpe={val_sharpe:.3f}"
            )

    return final_results


# ---------------------------------------------------------------------------
# 输出保存
# ---------------------------------------------------------------------------
def results_to_dataframe(results: List[BacktestResult]) -> pd.DataFrame:
    """将 BacktestResult 列表转为 DataFrame"""
    rows = []
    for r in results:
        p = r.param
        rows.append({
            "lookback": p.lookback,
            "obs_window": p.obs_window,
            "vol_lb": p.vol_lb,
            "vol_mul": p.vol_mul,
            "to_thresh": p.to_thresh,
            "holding_period": p.holding_period,
            "annual_return": r.annual_return,
            "sharpe_ratio": r.sharpe_ratio,
            "max_drawdown": r.max_drawdown,
            "annual_volatility": r.annual_volatility,
            "win_rate": r.win_rate,
            "trade_count": r.trade_count,
            "robustness_score": r.robustness,
            "val_annual_return": r.val_return,
            "val_sharpe_ratio": r.val_sharpe,
            "val_max_drawdown": r.val_maxdd,
            "eliminated_reason": r.eliminated_reason,
        })
    return pd.DataFrame(rows)


def save_results_csv(results: List[BacktestResult], output_path: Path):
    """保存结果到 CSV"""
    df = results_to_dataframe(results)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved results to {output_path}")


def save_best_params(results: List[BacktestResult], output_dir: Path, top_n: int = 3):
    """保存 Top N 稳健组合到 JSON"""
    top = sorted(results, key=lambda x: x.val_sharpe, reverse=True)[:top_n]
    best = []
    for r in top:
        best.append({
            "param": asdict(r.param),
            "train_sharpe_ratio": r.sharpe_ratio,
            "train_max_drawdown": r.max_drawdown,
            "train_win_rate": r.win_rate,
            "val_sharpe_ratio": r.val_sharpe,
            "val_max_drawdown": r.val_maxdd,
            "robustness_score": r.robustness,
        })
    out_path = output_dir / "best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved best params to {out_path}")


def plot_heatmap(results: List[BacktestResult], output_dir: Path):
    """生成 volume_multiplier × turnover_threshold 的验证集夏普热力图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping heatmap.")
        return

    rows = []
    for r in results:
        if r.eliminated_reason == "" and r.val_sharpe != 0:
            rows.append({
                "volume_multiplier": r.param.vol_mul,
                "turnover_threshold": r.param.to_thresh,
                "val_sharpe": r.val_sharpe,
            })

    if not rows:
        logger.warning("No valid results for heatmap.")
        return

    df = pd.DataFrame(rows)
    pivot = df.groupby(["volume_multiplier", "turnover_threshold"])["val_sharpe"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.1f}" for y in pivot.index])
    ax.set_xlabel("Turnover Threshold (%)")
    ax.set_ylabel("Volume Multiplier")
    ax.set_title("Validation Sharpe Ratio Heatmap")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax, label="Validation Sharpe")
    plt.tight_layout()
    out_path = output_dir / "param_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap to {out_path}")


# ---------------------------------------------------------------------------
# 最终测试集验收（Out-of-Sample）
# ---------------------------------------------------------------------------
def final_test_evaluation(
    top_results: List[BacktestResult],
    df_test_precomputed: pd.DataFrame,
    account: float,
    output_dir: Path,
) -> List[Dict]:
    """
    测试集 2024 数据仅在最终验收时使用，优化过程中绝对不可偷看。
    如果测试集夏普比验证集低 50% 以上，在输出中标注："警告：可能存在过拟合"。
    """
    test_results = []
    for r in top_results:
        res = run_single_backtest(r.param, df_test_precomputed, TEST_START, TEST_END, account)
        warning = ""
        if r.val_sharpe != 0 and res.sharpe_ratio < 0.5 * r.val_sharpe:
            warning = "警告：可能存在过拟合"
        test_results.append({
            "param": asdict(r.param),
            "test_annual_return": res.annual_return,
            "test_sharpe_ratio": res.sharpe_ratio,
            "test_max_drawdown": res.max_drawdown,
            "test_annual_volatility": res.annual_volatility,
            "test_win_rate": res.win_rate,
            "test_trade_count": res.trade_count,
            "val_sharpe_ratio": r.val_sharpe,
            "warning": warning,
        })

    out_path = output_dir / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("最终测试集验收结果（Out-of-Sample）")
    logger.info("=" * 60)
    for tr in test_results:
        logger.info(
            f"Test Sharpe={tr['test_sharpe_ratio']:.3f} | "
            f"Val Sharpe={tr['val_sharpe_ratio']:.3f} | "
            f"{tr['warning']}"
        )
    logger.info("=" * 60)
    return test_results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Golden Triangle Parameter Optimizer")
    p.add_argument("--data-path", required=True, help="本地 CSV 路径，需含 OHLCV+turnover 列")
    p.add_argument(
        "--output-dir",
        default="examples/golden_triangle/optimization_output",
        help="优化结果输出目录",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=min(8, multiprocessing.cpu_count()),
        help="并行进程数",
    )
    p.add_argument("--account", type=float, default=1_000_000, help="初始资金")
    p.add_argument("--top-k", type=int, default=20, help="训练集取 Top K 做稳健性测试")
    return p.parse_args()


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    df = load_data(args.data_path)

    # 2. 预计算训练集、验证集、测试集
    # 注意：测试集 2024 数据仅在最终验收时使用，优化过程中绝对不可偷看
    base_selector = EnhancedGoldenTriangleSelector()

    logger.info("Precomputing training set ...")
    df_train_precomputed = base_selector.precompute(slice_with_buffer(df, TRAIN_START, TRAIN_END))

    logger.info("Precomputing validation set ...")
    df_val_precomputed = base_selector.precompute(slice_with_buffer(df, VAL_START, VAL_END))

    logger.info("Precomputing test set (Out-of-Sample, DO NOT PEEK) ...")
    df_test_precomputed = base_selector.precompute(slice_with_buffer(df, TEST_START, TEST_END))

    # 3. 参数网格（lookback 和 volume_lookback 固定）
    param_grid = {
        "volume_multiplier": [1.2, 1.3, 1.5, 1.8, 2.0, 2.5],
        "turnover_threshold": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "observation_window": [2, 3, 5],
        "holding_period": [3, 5, 7, 10],
    }

    param_sets = [
        ParamSet(
            lookback=35,
            obs_window=ow,
            vol_lb=5,
            vol_mul=vm,
            to_thresh=tt,
            holding_period=hp,
        )
        for vm, tt, ow, hp in itertools.product(
            param_grid["volume_multiplier"],
            param_grid["turnover_threshold"],
            param_grid["observation_window"],
            param_grid["holding_period"],
        )
    ]
    logger.info(f"Total grid combinations: {len(param_sets)}")

    # 4. 训练集网格搜索
    logger.info("Step 1/4: Grid search on training set ...")
    train_args = [
        (p, df_train_precomputed, TRAIN_START, TRAIN_END, args.account)
        for p in param_sets
    ]
    train_results = parallel_run(_run_single_backtest_tuple, train_args, args.n_jobs)

    # 5. 稳健性测试（Top K）
    logger.info(f"Step 2/4: Robustness test on top-{args.top_k} training combos ...")
    top_train = sorted(train_results, key=lambda x: x.sharpe_ratio, reverse=True)[:args.top_k]
    robust_results = robustness_test(
        top_train, df_train_precomputed, args.account, args.n_jobs
    )
    logger.info(f"{len(robust_results)} / {len(top_train)} passed robustness test")

    # 6. 验证集盲测
    logger.info("Step 3/4: Validation set blind test ...")
    final_results = validation_test(robust_results, df_val_precomputed, args.account, args.n_jobs)
    logger.info(f"{len(final_results)} / {len(robust_results)} passed validation filters")

    # 7. 保存最终结果
    save_results_csv(final_results, output_dir / "optimization_results.csv")
    save_best_params(final_results, output_dir, top_n=3)
    plot_heatmap(final_results, output_dir)

    # 8. 最终测试集验收（绝对不可在优化过程中查看）
    logger.info("Step 4/4: Final test set evaluation (Out-of-Sample) ...")
    final_test_evaluation(final_results[:3], df_test_precomputed, args.account, output_dir)

    logger.info(f"All results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
