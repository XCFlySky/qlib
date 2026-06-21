# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Enhanced Golden Triangle Stock Selector（三状态互斥选股器）

在原有 GoldenTriangleSelector 的“有效买托”基础上，扩展为三层互斥信号：

1. Confirmed（右侧确认）
   MA5>MA10>MA20 多头排列，且观察窗口内 5穿10、10穿20 均已发生，
   同时放量、换手达标 → 视为最强信号，建议满仓买入。

2. Predicting（左侧埋伏）
   5穿10 已在观察窗口内发生，但 10穿20 尚未发生；
   MA10/MA20 ≥ predict_ma_ratio（默认 0.985，即距离 <1.5%），
   且 MA10 斜率 > MA20 斜率（slope_diff>0，说明 10 线追赶更快），
   量比 ≥ predict_vol_ratio（默认 1.3），换手 ≥ predict_turnover（默认 2.5%）
   → 半仓试探买入。

3. Forming（酝酿观察）
   5穿10 已在观察窗口内发生，10穿20 尚未发生；
   MA10/MA20 ≥ forming_ma_ratio（默认 0.97，即距离 <3%），
   量比 ≥ forming_vol_ratio（默认 1.1），换手 ≥ forming_turnover（默认 2.0%）
   → 仅加入观察池，不买入。

互斥规则：
- 优先级 Confirmed > Predicting > Forming。
- 同一只股票在同一交易日只能被打上一种标签。
- select() 支持 mode 参数：'all' | 'confirmed' | 'predicting' | 'forming'。
"""

from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger


class EnhancedGoldenTriangleSelector:
    """
    三状态互斥的有效买托选股器。

    Parameters
    ----------
    lookback_days : int
        每只股票加载的历史天数（默认 35）。
    observation_window : int
        扫描金叉信号的最近天数（默认 3，即 T、T-1、T-2）。
    volume_lookback : int
        计算量比时参照的前 N 个交易日均量（默认 5）。
    volume_multiplier : float
        Confirmed 状态的放量阈值（默认 1.5）。
    turnover_threshold : float
        Confirmed 状态的最低换手率，单位为百分比（默认 3.0）。
    predict_ma_ratio : float
        Predicting 状态允许 MA10/MA20 的最低比值（默认 0.985）。
    predict_vol_ratio : float
        Predicting 状态的最低量比（默认 1.3）。
    predict_turnover : float
        Predicting 状态的最低换手率，单位为百分比（默认 2.5）。
    forming_ma_ratio : float
        Forming 状态允许 MA10/MA20 的最低比值（默认 0.97）。
    forming_vol_ratio : float
        Forming 状态的最低量比（默认 1.1）。
    forming_turnover : float
        Forming 状态的最低换手率，单位为百分比（默认 2.0）。
    min_listing_days : int
        排除上市天数少于该值的股票（默认 20）。
    """

    def __init__(
        self,
        lookback_days: int = 35,
        observation_window: int = 3,
        volume_lookback: int = 5,
        volume_multiplier: float = 1.5,
        turnover_threshold: float = 3.0,
        predict_ma_ratio: float = 0.985,
        predict_vol_ratio: float = 1.3,
        predict_turnover: float = 2.5,
        forming_ma_ratio: float = 0.97,
        forming_vol_ratio: float = 1.1,
        forming_turnover: float = 2.0,
        min_listing_days: int = 20,
    ):
        self.lookback_days = lookback_days
        self.observation_window = observation_window
        self.volume_lookback = volume_lookback
        self.volume_multiplier = volume_multiplier
        self.turnover_threshold = turnover_threshold

        self.predict_ma_ratio = predict_ma_ratio
        self.predict_vol_ratio = predict_vol_ratio
        self.predict_turnover = predict_turnover
        self.forming_ma_ratio = forming_ma_ratio
        self.forming_vol_ratio = forming_vol_ratio
        self.forming_turnover = forming_turnover

        self.min_listing_days = min_listing_days

        # 缓存：在回测循环中避免对同一份 df 重复 precompute / pre_filter
        self._precomputed_key: Optional[int] = None
        self._precomputed_df: Optional[pd.DataFrame] = None
        self._prefilter_key: Optional[tuple] = None
        self._prefilter_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def precompute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        向量化预计算所有技术指标与金叉信号。

        计算字段：
        - ma5 / ma10 / ma20
        - cross_5_10 / cross_10_20
        - bull_arrange
        - avg_volume_N / volume_ratio
        - ma10_20_ratio
        - ma10_slope / ma20_slope / slope_diff
        - has_cross_5_10 / has_cross_10_20 / golden_triangle
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
        mode: str = "all",
        industry_map: Optional[pd.Series] = None,
        name_map: Optional[pd.Series] = None,
        st_set: Optional[set] = None,
    ) -> pd.DataFrame:
        """
        按三层互斥逻辑打标签并选股。

        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex（datetime, instrument）或包含 instrument 列。
            所需列（不区分大小写）：close / $close、volume / $volume、
            turnover / $turnover（换手率，单位为百分比）。
        trade_date : str or pd.Timestamp, optional
            观察基准日 T；为 None 时使用 df 中最新日期。
        mode : {'all', 'confirmed', 'predicting', 'forming'}
            返回的信号类型，默认 'all' 返回全部三层互斥结果。
        industry_map : pd.Series, optional
            instrument -> 行业名称 的映射。
        name_map : pd.Series, optional
            instrument -> 股票名称 的映射。
        st_set : set, optional
            要排除的 ST 股票代码集合。

        Returns
        -------
        pd.DataFrame
            每行一只股票，包含 signal_type 列（confirmed/predicting/forming）
            以及 cross_date、ma5、ma10、ma20、volume_ratio、turnover 等字段。
        """
        if mode not in {"all", "confirmed", "predicting", "forming"}:
            raise ValueError(f"Unsupported mode: {mode!r}")

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

        # 步骤 0：排除 ST / 停牌 / 新股
        df = self._pre_filter(df, st_set)
        if df.empty:
            return pd.DataFrame()

        # 如未预计算，则一次性计算全部信号
        if not self._is_precomputed(df):
            df = self.precompute(df)

        # 获取每个股票截至 T 的最新快照
        snapshot = self._get_snapshot(df, trade_date)
        if snapshot.empty:
            logger.warning(f"No snapshot available up to {trade_date.date()}.")
            return pd.DataFrame()

        # 计算观察窗口内及截至 T 的金叉标志
        cross_flags = self._compute_cross_flags(df, trade_date)
        snapshot = snapshot.join(cross_flags, how="left")

        # 三层互斥筛选（注意优先级）
        confirmed = self._filter_confirmed(snapshot)
        predicting = self._filter_predicting(snapshot)
        forming = self._filter_forming(snapshot)

        # ---- 互斥：高优先级状态覆盖低优先级状态 ----
        confirmed_insts = set(confirmed.index)
        predicting = predicting[~predicting.index.isin(confirmed_insts)]
        predicting_insts = set(predicting.index)
        forming = forming[~forming.index.isin(confirmed_insts | predicting_insts)]

        # 补充 cross_date：Confirmed 取窗口内首次 golden_triangle；
        # Predicting / Forming 取窗口内首次 cross_5_10。
        if not confirmed.empty:
            confirmed["cross_date"] = self._get_first_event_date(
                df, trade_date, "golden_triangle", confirmed.index
            )
        if not predicting.empty:
            predicting["cross_date"] = self._get_first_event_date(
                df, trade_date, "cross_5_10", predicting.index
            )
        if not forming.empty:
            forming["cross_date"] = self._get_first_event_date(
                df, trade_date, "cross_5_10", forming.index
            )

        confirmed["signal_type"] = "confirmed"
        predicting["signal_type"] = "predicting"
        forming["signal_type"] = "forming"

        result = pd.concat([confirmed, predicting, forming])
        if result.empty:
            logger.info("No stocks matched any signal state.")
            return pd.DataFrame()

        # 按 mode 过滤
        if mode != "all":
            result = result[result["signal_type"] == mode]

        logger.info(
            f"Signal counts: Confirmed={len(confirmed)}, "
            f"Predicting={len(predicting)}, Forming={len(forming)}"
        )

        result = self._attach_meta(result, industry_map, name_map)
        return result

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名并确保为 MultiIndex（datetime, instrument）。"""
        df = df.copy()

        # 如有必要，展平 MultiIndex 列
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                " ".join(col).strip() if col[1] not in ["nan", "NaN"] else col[0]
                for col in df.columns.values
            ]

        col_map = {}
        for c in df.columns:
            col_map[c] = str(c).lower().replace("$", "")
        df = df.rename(columns=col_map)

        required = {"close", "volume", "turnover"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")

        # 确保 MultiIndex（datetime, instrument）
        if "instrument" in df.columns:
            df = df.set_index(["datetime", "instrument"])
        elif not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                "DataFrame must have MultiIndex (datetime, instrument) or a column named 'instrument'."
            )

        # 仅保留所需列（保留 OHLC 供回测引擎按开盘价/收盘价成交）
        keep = ["open", "high", "low", "close", "volume", "turnover"]
        df = df[[c for c in keep if c in df.columns]]

        df = df.sort_index(level=["datetime", "instrument"])
        return df

    def _is_precomputed(self, df: pd.DataFrame) -> bool:
        """检查 DataFrame 是否已经包含预计算的信号列。"""
        required = {
            "ma5",
            "ma10",
            "ma20",
            "cross_5_10",
            "cross_10_20",
            "bull_arrange",
            f"avg_volume_{self.volume_lookback}",
            "volume_ratio",
            "ma10_20_ratio",
            "ma10_slope",
            "ma20_slope",
            "slope_diff",
            "golden_triangle",
        }
        return required.issubset(set(df.columns))

    def _pre_filter(self, df: pd.DataFrame, st_set: Optional[set]) -> pd.DataFrame:
        """排除 ST、停牌、新股（带缓存）。"""
        key = (id(df), id(st_set), len(st_set) if st_set is not None else None)
        if key == self._prefilter_key and self._prefilter_df is not None:
            return self._prefilter_df

        if st_set is not None:
            mask = ~df.index.get_level_values("instrument").isin(st_set)
            df = df[mask]

        # 排除窗口期内数据天数过少的股票
        counts = df.groupby(level="instrument").size()
        valid_insts = counts[counts >= self.min_listing_days].index
        df = df.loc[df.index.get_level_values("instrument").isin(valid_insts)]

        # 排除停牌股（最新一日成交量为 0）
        latest = df.groupby(level="instrument")["volume"].tail(1)
        suspended = latest[latest == 0].index.get_level_values("instrument").unique()
        if len(suspended):
            df = df.loc[~df.index.get_level_values("instrument").isin(suspended)]

        self._prefilter_key = key
        self._prefilter_df = df
        return df

    def _calc_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次性对所有股票计算技术指标（向量化 groupby）。"""

        def _calc(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_index(level="datetime")
            close = group["close"]
            vol = group["volume"]

            # 均线
            group["ma5"] = close.rolling(window=5, min_periods=5).mean()
            group["ma10"] = close.rolling(window=10, min_periods=10).mean()
            group["ma20"] = close.rolling(window=20, min_periods=20).mean()

            # 金叉信号
            group["cross_5_10"] = (group["ma5"] > group["ma10"]) & (
                group["ma5"].shift(1) <= group["ma10"].shift(1)
            )
            group["cross_10_20"] = (group["ma10"] > group["ma20"]) & (
                group["ma10"].shift(1) <= group["ma20"].shift(1)
            )

            # 多头排列
            group["bull_arrange"] = (group["ma5"] > group["ma10"]) & (group["ma10"] > group["ma20"])

            # 量比：当日量 / 前 N 日平均量
            col = f"avg_volume_{self.volume_lookback}"
            group[col] = (
                vol.shift(1).rolling(window=self.volume_lookback, min_periods=self.volume_lookback).mean()
            )
            group["volume_ratio"] = group["volume"] / group[col]

            # MA10 与 MA20 的贴近程度
            group["ma10_20_ratio"] = group["ma10"] / group["ma20"]

            # 斜率：近 5 日线性变化速率
            group["ma10_slope"] = (group["ma10"] - group["ma10"].shift(5)) / 5
            group["ma20_slope"] = (group["ma20"] - group["ma20"].shift(5)) / 5
            group["slope_diff"] = group["ma10_slope"] - group["ma20_slope"]

            return group

        df = df.groupby(level="instrument", group_keys=False).apply(_calc)
        return df

    def _mark_triangle_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次性标记所有股票在所有交易日的金叉三角形累积信号。"""

        def _mark(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_index(level="datetime")
            group["has_cross_5_10"] = group["cross_5_10"].cumsum() > 0
            group["has_cross_10_20"] = group["cross_10_20"].cumsum() > 0
            group["golden_triangle"] = (
                group["has_cross_5_10"] & group["has_cross_10_20"] & group["bull_arrange"]
            )
            return group

        df = df.groupby(level="instrument", group_keys=False).apply(_mark)
        return df

    def _get_snapshot(self, df: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
        """取每个股票截至 trade_date 的最新一行，索引设为 instrument。"""
        mask = df.index.get_level_values("datetime") <= trade_date
        snapshot = df[mask].groupby(level="instrument").tail(1).copy()
        if snapshot.empty:
            return snapshot
        snapshot = snapshot.reset_index(level="datetime")
        return snapshot

    def _compute_cross_flags(
        self, df: pd.DataFrame, trade_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        计算每个股票在观察窗口内及截至 T 的金叉标志：
        - cross_5_10_in_window：观察窗口内是否发生 5穿10
        - cross_10_20_in_window：观察窗口内是否发生 10穿20
        - cross_10_20_ever：截至 T 是否发生过 10穿20（用于排除 Predicting / Forming）
        """
        all_dates = df.index.get_level_values("datetime").unique().sort_values()
        obs_dates = all_dates[all_dates <= trade_date]
        if len(obs_dates) < self.observation_window:
            logger.warning(
                f"Only {len(obs_dates)} trading days available up to {trade_date.date()}, "
                f"need at least {self.observation_window} for observation window."
            )
            return pd.DataFrame()
        obs_dates = obs_dates[-self.observation_window :]

        df_obs = df[df.index.get_level_values("datetime").isin(obs_dates)]
        df_upto = df[df.index.get_level_values("datetime") <= trade_date]

        cross_flags = pd.DataFrame(
            {
                "cross_5_10_in_window": df_obs.groupby(level="instrument")["cross_5_10"].any(),
                "cross_10_20_in_window": df_obs.groupby(level="instrument")["cross_10_20"].any(),
                "cross_10_20_ever": df_upto.groupby(level="instrument")["cross_10_20"].any(),
            }
        )
        return cross_flags

    def _filter_confirmed(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """
        Confirmed（右侧确认）：
        观察窗口内 5穿10、10穿20 均已发生，且当前 MA5>MA10>MA20，
        放量、换手达标。
        """
        mask = (
            snapshot["bull_arrange"].fillna(False)
            & snapshot["cross_5_10_in_window"].fillna(False)
            & snapshot["cross_10_20_in_window"].fillna(False)
            & (snapshot["volume_ratio"] >= self.volume_multiplier)
            & (snapshot["turnover"] >= self.turnover_threshold)
        )
        return snapshot[mask].copy()

    def _filter_predicting(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """
        Predicting（左侧埋伏）：
        5穿10 已在观察窗口内发生，10穿20 截至当前尚未发生；
        MA10/MA20 ≥ predict_ma_ratio，slope_diff>0，量比/换手达标。
        """
        mask = (
            snapshot["cross_5_10_in_window"].fillna(False)
            & ~snapshot["cross_10_20_ever"].fillna(False)
            & (snapshot["ma10_20_ratio"] >= self.predict_ma_ratio)
            & (snapshot["slope_diff"] > 0)
            & (snapshot["volume_ratio"] >= self.predict_vol_ratio)
            & (snapshot["turnover"] >= self.predict_turnover)
        )
        return snapshot[mask].copy()

    def _filter_forming(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """
        Forming（酝酿观察）：
        5穿10 已在观察窗口内发生，10穿20 截至当前尚未发生；
        MA10/MA20 ≥ forming_ma_ratio，量比/换手达标。
        本层条件最宽松，仅入观察池。
        """
        mask = (
            snapshot["cross_5_10_in_window"].fillna(False)
            & ~snapshot["cross_10_20_ever"].fillna(False)
            & (snapshot["ma10_20_ratio"] >= self.forming_ma_ratio)
            & (snapshot["volume_ratio"] >= self.forming_vol_ratio)
            & (snapshot["turnover"] >= self.forming_turnover)
        )
        return snapshot[mask].copy()

    def _get_first_event_date(
        self,
        df: pd.DataFrame,
        trade_date: pd.Timestamp,
        event_col: str,
        instruments: pd.Index,
    ) -> pd.Series:
        """返回每只股票在观察窗口内首次 event_col=True 的日期。"""
        all_dates = df.index.get_level_values("datetime").unique().sort_values()
        obs_dates = all_dates[all_dates <= trade_date]
        if len(obs_dates) < self.observation_window:
            return pd.Series(index=instruments, dtype="datetime64[ns]")
        obs_dates = obs_dates[-self.observation_window :]

        df_obs = df[
            df.index.get_level_values("datetime").isin(obs_dates)
            & df.index.get_level_values("instrument").isin(instruments)
        ].copy()
        df_obs = df_obs.reset_index()
        event_rows = df_obs[df_obs[event_col]]
        first_dates = event_rows.groupby("instrument")["datetime"].first()
        return first_dates.reindex(instruments)

    def _attach_meta(
        self,
        df: pd.DataFrame,
        industry_map: Optional[pd.Series],
        name_map: Optional[pd.Series],
    ) -> pd.DataFrame:
        """附加名称/行业/信号类型信息并生成最终列布局。"""
        inst = df.index.get_level_values("instrument")

        df["instrument"] = inst
        df["name"] = inst.map(name_map) if name_map is not None else np.nan
        df["industry"] = inst.map(industry_map) if industry_map is not None else np.nan

        col = f"avg_volume_{self.volume_lookback}"

        # 输出列顺序：先标识与信号类型，再技术指标，再成交量/换手
        cols = [
            "instrument",
            "name",
            "signal_type",
            "cross_date",
            "ma5",
            "ma10",
            "ma20",
            "volume",
            col,
            "volume_ratio",
            "turnover",
            "ma10_20_ratio",
            "slope_diff",
            "industry",
        ]
        cols = [c for c in cols if c in df.columns]
        result = df[cols].copy()
        result = result.reset_index(drop=True)
        return result
