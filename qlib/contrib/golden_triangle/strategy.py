# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Golden Triangle Backtest Strategy
==================================

Integrates the Golden Triangle stock selector with Qlib's backtest engine.

Usage example::

    from qlib.contrib.golden_triangle.strategy import GoldenTriangleStrategy

    strategy = GoldenTriangleStrategy(
        provider_uri="~/.qlib/qlib_data/cn_data",
        turnover_source="qlib",
        max_positions=10,
        holding_period=5,
    )
"""

from typing import Optional, List, Set
import pandas as pd
from loguru import logger

from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO

from .selector import GoldenTriangleSelector
from .data_source import HybridDataSource, TurnoverFetcher


class GoldenTriangleStrategy(BaseStrategy):
    """
    Event-driven backtest strategy for the Golden Triangle selector.

    Logic:
    1. On each trading day, run the Golden Triangle selector using data up to
       the previous close.
    2. Sell positions whose holding-period has expired or which are no longer
       selected (configurable).
    3. Distribute available cash equally among newly selected stocks, subject
       to ``max_positions``.

    Parameters
    ----------
    provider_uri : str, optional
        Qlib data directory.  If None, uses the already-initialized Qlib config.
    region : str, default "cn"
        Qlib region.
    turnover_source : str, default "qlib"
        Where to load turnover from (see :class:`HybridDataSource`).
    turnover_kwargs : dict, optional
        Extra kwargs forwarded to the turnover fetcher.
    max_positions : int, default 10
        Maximum number of stocks held simultaneously.
    holding_period : int, default 5
        Minimum number of trading days to hold a newly bought position.
    sell_on_exit_signal : bool, default False
        If True, sell a position as soon as it disappears from the selection set,
        even before ``holding_period`` expires.
    rebalance_freq : str, default "day"
        Rebalancing frequency.  Currently only ``"day"`` is supported.
    risk_degree : float, default 0.95
        Fraction of total account value that can be invested in stocks.
    industry_filter : list of str, optional
        Only keep selected stocks whose industry contains one of these keywords.
    filter_st : bool, default True
        Whether to filter out ST stocks using akshare (requires network).
    lookback_days, observation_window, volume_lookback, volume_multiplier,
    turnover_threshold, min_listing_days :
        Passed to :class:`GoldenTriangleSelector`.
    """

    def __init__(
        self,
        provider_uri: Optional[str] = None,
        region: str = "cn",
        turnover_source: str = "qlib",
        turnover_kwargs: Optional[dict] = None,
        max_positions: int = 10,
        holding_period: int = 5,
        sell_on_exit_signal: bool = False,
        rebalance_freq: str = "day",
        risk_degree: float = 0.95,
        industry_filter: Optional[List[str]] = None,
        filter_st: bool = True,
        lookback_days: int = 35,
        observation_window: int = 3,
        volume_lookback: int = 5,
        volume_multiplier: float = 1.5,
        turnover_threshold: float = 3.0,
        min_listing_days: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._data_source = HybridDataSource(
            provider_uri=provider_uri,
            region=region,
            turnover_source=turnover_source,
            turnover_kwargs=turnover_kwargs or {},
        )
        self._selector = GoldenTriangleSelector(
            lookback_days=lookback_days,
            observation_window=observation_window,
            volume_lookback=volume_lookback,
            volume_multiplier=volume_multiplier,
            turnover_threshold=turnover_threshold,
            min_listing_days=min_listing_days,
        )
        self.max_positions = max_positions
        self.holding_period = holding_period
        self.sell_on_exit_signal = sell_on_exit_signal
        self.rebalance_freq = rebalance_freq
        self.risk_degree = risk_degree
        self.industry_filter = [k.strip() for k in industry_filter] if industry_filter else None
        self.filter_st = filter_st

        self._stock_info: Optional[pd.DataFrame] = None
        self._st_set: Optional[Set[str]] = None
        self._ready = False

    def _prepare_static_info(self):
        """Load stock names / industries and ST list once."""
        if self._ready:
            return
        try:
            self._stock_info = self._data_source.fetch_stock_info()
        except Exception as e:
            logger.warning(f"Could not fetch stock info: {e}")
            self._stock_info = None

        if self.filter_st:
            try:
                self._st_set = set(TurnoverFetcher.fetch_st_list())
                logger.info(f"Fetched {len(self._st_set)} ST stocks for exclusion.")
            except Exception as e:
                logger.warning(f"Could not fetch ST list: {e}")
                self._st_set = None
        self._ready = True

    def _is_rebalance_day(self, trade_step: int, trade_date: pd.Timestamp) -> bool:
        if self.rebalance_freq == "day":
            return True
        # Weekly (every Monday) can be added here when needed.
        return True

    def _select_stocks(self, signal_date: pd.Timestamp) -> pd.DataFrame:
        """Run the selector using data up to ``signal_date`` (usually the previous close)."""
        lookback = self._selector.lookback_days + 10
        start_date = (signal_date - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
        end_date = signal_date.strftime("%Y-%m-%d")

        # Determine instrument universe from the exchange if available.
        exchange = self.trade_exchange
        if exchange is not None and hasattr(exchange, "codes"):
            instruments = exchange.codes
        else:
            instruments = "all"

        df = self._data_source.fetch(instruments, start_date, end_date)

        name_map = (
            self._stock_info["name"]
            if self._stock_info is not None and "name" in self._stock_info.columns
            else None
        )
        industry_map = (
            self._stock_info["industry"]
            if self._stock_info is not None and "industry" in self._stock_info.columns
            else None
        )

        result = self._selector.select(
            df,
            trade_date=signal_date,
            industry_map=industry_map,
            name_map=name_map,
            st_set=self._st_set,
        )

        if self.industry_filter and not result.empty:
            keywords = self.industry_filter
            mask = result["industry"].astype(str).apply(lambda x: any(k in x for k in keywords))
            result = result[mask].copy()

        return result

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start, trade_end = self.trade_calendar.get_step_time(trade_step)
        trade_date = pd.Timestamp(trade_start.date())

        self._prepare_static_info()

        if not self._is_rebalance_day(trade_step, trade_date):
            return TradeDecisionWO([], self)

        # The selector uses information up to the previous close, so run it on
        # the previous trading day.  If we are at the very first bars and don't
        # have enough history, skip.
        if trade_step < self._selector.min_listing_days:
            return TradeDecisionWO([], self)

        try:
            # ``shift=1`` gives the interval of the previous bar.
            prev_end = self.trade_calendar.get_step_time(trade_step, shift=1)[1]
            signal_date = pd.Timestamp(prev_end.date())
            selected = self._select_stocks(signal_date)
        except Exception as e:
            logger.warning(f"Selector failed on {trade_date.date()}: {e}")
            return TradeDecisionWO([], self)

        selected_insts = selected["instrument"].tolist() if not selected.empty else []
        logger.info(f"[{trade_date.date()}] Selected {len(selected_insts)} stocks")

        position = self.trade_position
        current = position.get_stock_amount_dict()
        # Remove cash key if present.
        current = {k: v for k, v in current.items() if k != "cash"}

        orders = []

        # ---- Sell decisions -------------------------------------------------
        for inst, amount in current.items():
            if amount <= 0:
                continue

            held_days = position.get_stock_count(inst, self.rebalance_freq)
            expired = held_days >= self.holding_period
            exit_signal = self.sell_on_exit_signal and inst not in selected_insts

            if expired or exit_signal:
                # Check tradability.
                if self.trade_exchange.is_stock_tradable(inst, trade_start, trade_end):
                    orders.append(
                        Order(
                            stock_id=inst,
                            amount=amount,
                            direction=OrderDir.SELL,
                            start_time=trade_start,
                            end_time=trade_end,
                        )
                    )
                    logger.info(f"  SELL {inst} amount={amount:.0f} (held={held_days}d)")

        # ---- Buy decisions --------------------------------------------------
        # Recalculate available slots after accounting for pending sells.
        remaining_current = {
            inst: amt for inst, amt in current.items()
            if amt > 0 and inst not in [o.stock_id for o in orders if o.direction == OrderDir.SELL]
        }
        slots_available = self.max_positions - len(remaining_current)

        if slots_available <= 0:
            return TradeDecisionWO(orders, self)

        # Candidates: selected stocks we don't already hold.
        held_set = set(remaining_current.keys())
        candidates = [inst for inst in selected_insts if inst not in held_set]
        candidates = candidates[:slots_available]

        if not candidates:
            return TradeDecisionWO(orders, self)

        # Use risk_degree to reserve a fraction of total value in cash.
        total_value = position.calculate_value()
        investable_value = total_value * self.risk_degree
        cash = position.get_cash()
        # Do not invest more than available cash.
        budget = min(cash, investable_value - sum(
            position.get_stock_amount(inst) * position.get_stock_price(inst)
            for inst in remaining_current
        ))
        budget_per_stock = budget / len(candidates) if candidates else 0.0

        for inst in candidates:
            if not self.trade_exchange.is_stock_tradable(inst, trade_start, trade_end):
                logger.info(f"  SKIP {inst} (not tradable)")
                continue

            price = self.trade_exchange.get_close(inst, trade_start, trade_end)
            if price is None or price <= 0 or pd.isna(price):
                logger.info(f"  SKIP {inst} (no price)")
                continue

            # Convert budget to share amount.  Round down to whole shares.
            amount = int(budget_per_stock / price)
            if amount <= 0:
                continue

            orders.append(
                Order(
                    stock_id=inst,
                    amount=amount,
                    direction=OrderDir.BUY,
                    start_time=trade_start,
                    end_time=trade_end,
                )
            )
            logger.info(f"  BUY {inst} amount={amount:.0f} @ ~{price:.2f}")

        return TradeDecisionWO(orders, self)
