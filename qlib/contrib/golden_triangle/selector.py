# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Golden Triangle Stock Selector (有效买托策略)

核心筛选逻辑：
1. 金叉三角形：MA5上穿MA10且MA10上穿MA20，
   在短窗口内形成多头排列（MA5 > MA10 > MA20）
2. 放量：交叉日成交量 >= 前7个交易日平均成交量的1.5倍
3. 换手率过滤：交叉日换手率 > 3%
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from loguru import logger


class GoldenTriangleSelector:
    """
    有效买托多因子选股器

    Parameters
    ----------
    lookback_days : int
        每只股票加载的历史天数（默认30）。
    observation_window : int
        扫描金叉信号的最近天数（默认3，即T、T-1、T-2）。
    volume_multiplier : float
        放量阈值（默认1.5）。
    turnover_threshold : float
        最低换手率，单位为百分比（默认3.0）。
    min_listing_days : int
        排除上市天数少于该值的股票（默认20）。
    """

    def __init__(
        self,
        lookback_days: int = 35,
        observation_window: int = 3,
        volume_lookback: int = 5,
        volume_multiplier: float = 1.5,
        turnover_threshold: float = 3.0,
        min_listing_days: int = 20,
    ):
        self.lookback_days = lookback_days
        self.observation_window = observation_window
        self.volume_lookback = volume_lookback
        self.volume_multiplier = volume_multiplier
        self.turnover_threshold = turnover_threshold
        self.min_listing_days = min_listing_days

        # Cache for repeated select() calls in a backtest loop
        self._precomputed_key: Optional[int] = None
        self._precomputed_df: Optional[pd.DataFrame] = None
        self._prefilter_key: Optional[int] = None
        self._prefilter_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一次性预计算所有技术指标和金叉信号。

        在回测循环中，可将本函数的结果传入 ``select()``，
        从而避免每个交易日重复 groupby/rolling 计算。
        """
        df = self._normalize_input(df)
        if df.empty:
            return df
        df = self._calc_all_indicators(df)
        df = self._mark_triangle_all(df)
        return df

    def select(
        self,
        df: pd.DataFrame,
        trade_date: Optional[str] = None,
        industry_map: Optional[pd.Series] = None,
        name_map: Optional[pd.Series] = None,
        st_set: Optional[set] = None,
    ) -> pd.DataFrame:
        """
        执行四步漏斗筛选。

        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex（日期时间，股票代码）或单索引日期时间，
            并包含 ``instrument`` 列。所需列（不区分大小写）：
            ``close`` / ``$close``、``volume`` / ``$volume``、
            ``turnover`` / ``$turnover``（换手率，单位为百分比）。
        trade_date : str or pd.Timestamp, optional
            观察基准日期 ``T``。若为None，则使用 ``df`` 中的最新日期。
        industry_map : pd.Series, optional
            以股票代码为索引的行业名称Series。
            例如 ``industry_map['SH600000'] = '银行'``
        name_map : pd.Series, optional
            以股票代码为索引的股票名称Series。
        st_set : set, optional
            要排除的ST股票代码集合。

        Returns
        -------
        pd.DataFrame
            每行代表一只入选股票，列包括：
            - instrument, name, cross_date, ma5, ma10, ma20,
            - volume, avg_volume_N, volume_ratio, turnover, industry
              （N 为 ``volume_lookback``）
        """
        df = self._normalize_input(df)
        if df.empty:
            logger.warning("Input DataFrame is empty after normalisation.")
            return pd.DataFrame()

        # 基准日期
        if trade_date is None:
            trade_date = df.index.get_level_values("datetime").max()
        else:
            trade_date = pd.Timestamp(trade_date)

        logger.info(f"Observation anchor date: {trade_date.date()}")

        # ---- 步骤0：排除ST/停牌/新股 ----------------
        df = self._pre_filter(df, st_set)
        if df.empty:
            return pd.DataFrame()

        # 如果输入尚未预计算，则一次性计算所有信号
        if not self._is_precomputed(df):
            df = self.precompute(df)

        # ---- 步骤1：金叉三角形 --------------------------------
        step1 = self._filter_golden_triangle(df, trade_date)
        logger.info(f"Step 1 (Golden Triangle): {len(step1)} stocks")
        if step1.empty:
            return pd.DataFrame()

        # ---- 步骤2：放量 -----------------------------------------
        step2 = self._filter_volume_surge(step1)
        logger.info(f"Step 2 (Volume Surge): {len(step2)} stocks")
        if step2.empty:
            return pd.DataFrame()

        # ---- 步骤3：换手率过滤 --------------------------------------
        step3 = self._filter_turnover(step2)
        logger.info(f"Step 3 (Turnover > {self.turnover_threshold}%): {len(step3)} stocks")
        if step3.empty:
            return pd.DataFrame()

        # ---- 步骤4：行业/名称/格式化 -----------------------------
        result = self._attach_meta(step3, industry_map, name_map)
        return result

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名并确保为MultiIndex。"""
        df = df.copy()

        # 如有必要，展平MultiIndex列
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(col).strip() if col[1] not in ["nan", "NaN"] else col[0] for col in df.columns.values]

        col_map = {}
        for c in df.columns:
            lower = str(c).lower().replace("$", "")
            col_map[c] = lower
        df = df.rename(columns=col_map)

        required = {"close", "volume", "turnover"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")

        # 确保MultiIndex（日期时间，股票代码）
        if "instrument" in df.columns:
            df = df.set_index(["datetime", "instrument"])
        elif not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame must have MultiIndex (datetime, instrument) or a column named 'instrument'.")

        # 仅保留所需列（保留 OHLC 供回测引擎按开盘价/收盘价成交）
        keep = ["open", "high", "low", "close", "volume", "turnover"]
        df = df[[c for c in keep if c in df.columns]]

        # 排序以便滚动计算
        df = df.sort_index(level=["datetime", "instrument"])
        return df

    def _is_precomputed(self, df: pd.DataFrame) -> bool:
        """检查 DataFrame 是否已经包含预计算的信号列。"""
        required = {"ma5", "ma10", "ma20", "cross_5_10", "cross_10_20",
                    "bull_arrange", f"avg_volume_{self.volume_lookback}", "golden_triangle"}
        return required.issubset(set(df.columns))

    def _pre_filter(
        self,
        df: pd.DataFrame,
        st_set: Optional[set],
    ) -> pd.DataFrame:
        """排除停牌、ST和新股（带缓存，避免回测循环中重复计算）。"""
        key = (id(df), id(st_set), len(st_set) if st_set is not None else None)
        if key == self._prefilter_key and self._prefilter_df is not None:
            return self._prefilter_df

        # 排除ST股票
        if st_set is not None:
            mask = ~df.index.get_level_values("instrument").isin(st_set)
            df = df[mask]

        # 排除窗口期内数据天数过少的股票
        counts = df.groupby(level="instrument").size()
        valid_insts = counts[counts >= self.min_listing_days].index
        df = df.loc[df.index.get_level_values("instrument").isin(valid_insts)]

        # 排除停牌股（各股票最新一日成交量为0）
        latest = df.groupby(level="instrument")["volume"].tail(1)
        suspended = latest[latest == 0].index.get_level_values("instrument").unique()
        if len(suspended):
            df = df.loc[~df.index.get_level_values("instrument").isin(suspended)]

        self._prefilter_key = key
        self._prefilter_df = df
        return df

    def _filter_golden_triangle(
        self,
        df: pd.DataFrame,
        trade_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        步骤1：在截至 ``trade_date`` 的观察窗口内，
        识别具有有效金叉三角形的股票。

        逻辑：在观察窗口内，MA5 金叉 MA10 与 MA10 金叉 MA20 两个事件
        只要都发生过（不分先后），且截至某日已形成多头排列，即判定有效。
        """
        # 构建观察日历：取截至trade_date的最后observation_window个交易日
        all_dates = df.index.get_level_values("datetime").unique().sort_values()
        obs_dates = all_dates[all_dates <= trade_date]
        if len(obs_dates) < self.observation_window:
            logger.warning(
                f"Only {len(obs_dates)} trading days available up to {trade_date.date()}, "
                f"need at least {self.observation_window}."
            )
            return pd.DataFrame()

        obs_dates = obs_dates[-self.observation_window :]

        # 如果输入已经预计算过，直接切片到观察窗口
        if self._is_precomputed(df):
            df_obs = df[df.index.get_level_values("datetime").isin(obs_dates)].copy()
        else:
            # 计算单只股票指标
            def _calc_indicators(group: pd.DataFrame) -> pd.DataFrame:
                group = group.sort_index(level="datetime")
                close = group["close"]
                vol = group["volume"]

                # 移动平均线
                group["ma5"] = close.rolling(window=5, min_periods=5).mean()
                group["ma10"] = close.rolling(window=10, min_periods=10).mean()
                group["ma20"] = close.rolling(window=20, min_periods=20).mean()

                # 交叉信号
                group["cross_5_10"] = (group["ma5"] > group["ma10"]) & (
                    group["ma5"].shift(1) <= group["ma10"].shift(1)
                )
                group["cross_10_20"] = (group["ma10"] > group["ma20"]) & (
                    group["ma10"].shift(1) <= group["ma20"].shift(1)
                )

                # 多头排列
                group["bull_arrange"] = (group["ma5"] > group["ma10"]) & (group["ma10"] > group["ma20"])

                # 成交量：每日前N日平均（严格前N天）
                group[f"avg_volume_{self.volume_lookback}"] = (
                    vol.shift(1).rolling(window=self.volume_lookback, min_periods=self.volume_lookback).mean()
                )

                return group

            df = df.groupby(level="instrument").apply(_calc_indicators)
            # 删除groupby.apply在MultiIndex上引入的额外instrument索引层级
            if isinstance(df.index, pd.MultiIndex) and df.index.nlevels > 2:
                df = df.droplevel(0)

            # 限制在观察日期范围内
            mask_obs = df.index.get_level_values("datetime").isin(obs_dates)
            df_obs = df[mask_obs].copy()

            if df_obs.empty:
                return pd.DataFrame()

            # 在观察窗口内做累积判断：截至当日，两个金叉是否都已发生
            def _mark_triangle(group: pd.DataFrame) -> pd.DataFrame:
                group = group.sort_index(level="datetime")
                group["has_cross_5_10"] = group["cross_5_10"].cumsum() > 0
                group["has_cross_10_20"] = group["cross_10_20"].cumsum() > 0
                group["golden_triangle"] = (
                    group["has_cross_5_10"] & group["has_cross_10_20"] & group["bull_arrange"]
                )
                return group

            df_obs = df_obs.groupby(level="instrument").apply(_mark_triangle)
            if isinstance(df_obs.index, pd.MultiIndex) and df_obs.index.nlevels > 2:
                df_obs = df_obs.droplevel(0)

        # 选取满足三角形条件的最近观察日期
        gt_true = df_obs[df_obs["golden_triangle"]]
        if gt_true.empty:
            return pd.DataFrame()

        # 对每只股票，取观察窗口内首次满足条件的那一行（即金叉触发日）
        cross_rows = gt_true.groupby(level="instrument").head(1).copy()
        cross_rows["cross_date"] = cross_rows.index.get_level_values("datetime")

        return cross_rows

    def _calc_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次性对所有股票计算技术指标（向量化 groupby）。"""
        def _calc(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_index(level="datetime")
            close = group["close"]
            vol = group["volume"]

            group["ma5"] = close.rolling(window=5, min_periods=5).mean()
            group["ma10"] = close.rolling(window=10, min_periods=10).mean()
            group["ma20"] = close.rolling(window=20, min_periods=20).mean()
            group["cross_5_10"] = (group["ma5"] > group["ma10"]) & (
                group["ma5"].shift(1) <= group["ma10"].shift(1)
            )
            group["cross_10_20"] = (group["ma10"] > group["ma20"]) & (
                group["ma10"].shift(1) <= group["ma20"].shift(1)
            )
            group["bull_arrange"] = (group["ma5"] > group["ma10"]) & (group["ma10"] > group["ma20"])
            group[f"avg_volume_{self.volume_lookback}"] = (
                vol.shift(1).rolling(window=self.volume_lookback, min_periods=self.volume_lookback).mean()
            )
            return group

        df = df.groupby(level="instrument").apply(_calc)
        if isinstance(df.index, pd.MultiIndex) and df.index.nlevels > 2:
            df = df.droplevel(0)
        return df

    def _mark_triangle_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次性标记所有股票在所有交易日的金叉三角形信号。"""
        def _mark(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_index(level="datetime")
            group["has_cross_5_10"] = group["cross_5_10"].cumsum() > 0
            group["has_cross_10_20"] = group["cross_10_20"].cumsum() > 0
            group["golden_triangle"] = (
                group["has_cross_5_10"] & group["has_cross_10_20"] & group["bull_arrange"]
            )
            return group

        df = df.groupby(level="instrument").apply(_mark)
        if isinstance(df.index, pd.MultiIndex) and df.index.nlevels > 2:
            df = df.droplevel(0)
        return df

    def _filter_volume_surge(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤2：交叉日成交量 >= 前N日平均的1.5倍。"""
        if df.empty:
            return df
        col = f"avg_volume_{self.volume_lookback}"
        valid = df["volume"] >= df[col] * self.volume_multiplier
        valid &= df[col].notna()
        return df[valid].copy()

    def _filter_turnover(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤3：换手率 > 阈值。"""
        if df.empty:
            return df
        # turnover预期为百分比（例如5.2表示5.2%）
        valid = df["turnover"] > self.turnover_threshold
        return df[valid].copy()

    def _attach_meta(
        self,
        df: pd.DataFrame,
        industry_map: Optional[pd.Series],
        name_map: Optional[pd.Series],
    ) -> pd.DataFrame:
        """附加名称/行业信息并生成最终列布局。"""
        inst = df.index.get_level_values("instrument")

        df["instrument"] = inst
        df["name"] = inst.map(name_map) if name_map is not None else np.nan
        df["industry"] = inst.map(industry_map) if industry_map is not None else np.nan

        col = f"avg_volume_{self.volume_lookback}"
        df["volume_ratio"] = df["volume"] / df[col]

        # 重新排序/选择列
        cols = [
            "instrument",
            "name",
            "cross_date",
            "ma5",
            "ma10",
            "ma20",
            "volume",
            f"avg_volume_{self.volume_lookback}",
            "volume_ratio",
            "turnover",
            "industry",
        ]
        # 仅保留存在的列
        cols = [c for c in cols if c in df.columns]
        result = df[cols].copy()
        result = result.reset_index(drop=True)
        return result
