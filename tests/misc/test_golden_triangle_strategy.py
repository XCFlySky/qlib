# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unit tests for the GoldenTriangleStrategy integration into Qlib's event-driven
backtest engine.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qlib.backtest.decision import OrderDir
from qlib.contrib.golden_triangle.strategy import GoldenTriangleStrategy


def _make_ohlcv_turnover(n_days: int = 30) -> pd.DataFrame:
    """Build a tiny MultiIndex dataset that triggers a golden-triangle signal."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = np.ones(n_days) * 10.0
    # Trigger the triangle on the last day
    close[-1] = 11.0

    volume = np.ones(n_days) * 10000.0
    volume[-1] = 30000.0  # volume surge

    turnover = np.ones(n_days) * 2.0
    turnover[-1] = 5.0  # > 3%

    df = pd.DataFrame(
        {
            "datetime": dates,
            "instrument": ["SH600000"] * n_days,
            "close": close,
            "volume": volume,
            "turnover": turnover,
        }
    ).set_index(["datetime", "instrument"])
    return df


class TestGoldenTriangleStrategy:
    @pytest.fixture
    def strategy(self, monkeypatch):
        strategy = GoldenTriangleStrategy(
            filter_st=False,
            max_positions=5,
            holding_period=3,
            sell_on_exit_signal=False,
        )

        # Monkey-patch the data source so no real network / Qlib data is needed.
        strategy._data_source.fetch = lambda instruments, start, end: _make_ohlcv_turnover()
        strategy._data_source.fetch_stock_info = lambda: pd.DataFrame()

        return strategy

    def _build_infra(self, strategy, position_state: dict, trade_step: int = 30, price: float = 10.0):
        """Attach fake position / exchange / calendar infrastructure to ``strategy``."""

        # ---- fake position -------------------------------------------------
        class FakePosition:
            def __init__(self, state):
                self.cash = state.get("cash", 1_000_000.0)
                self._positions = state.get("positions", {})

            def get_stock_amount_dict(self):
                return {code: info["amount"] for code, info in self._positions.items()}

            def get_stock_count(self, code, bar):
                return self._positions.get(code, {}).get("count", 0)

            def calculate_value(self):
                stock_value = sum(
                    info["amount"] * info.get("price", price)
                    for info in self._positions.values()
                )
                return self.cash + stock_value

            def get_cash(self, include_settle=False):
                return self.cash

            def get_stock_amount(self, code):
                return self._positions.get(code, {}).get("amount", 0)

            def get_stock_price(self, code):
                return self._positions.get(code, {}).get("price", price)

        position = FakePosition(position_state)

        # ---- fake account --------------------------------------------------
        account = SimpleNamespace(current_position=position)

        # ---- fake exchange -------------------------------------------------
        class FakeExchange:
            codes = ["SH600000"]

            def is_stock_tradable(self, stock_id, start, end):
                return True

            def get_close(self, stock_id, start, end):
                return price

        exchange = FakeExchange()

        # ---- fake calendar -------------------------------------------------
        today = pd.Timestamp("2024-02-13")
        prev_day = pd.Timestamp("2024-02-12")

        class FakeCalendar:
            def get_trade_step(self):
                return trade_step

            def get_step_time(self, step=None, shift=0):
                if shift == 1:
                    return (prev_day, prev_day + pd.Timedelta(hours=23, minutes=59))
                return (today, today + pd.Timedelta(hours=23, minutes=59))

        calendar = FakeCalendar()

        # ---- infra wrappers ------------------------------------------------
        level_infra = SimpleNamespace(
            get=lambda name: calendar if name == "trade_calendar" else None,
        )
        common_infra = SimpleNamespace(
            get=lambda name: account if name == "trade_account" else exchange if name == "trade_exchange" else None,
        )

        strategy.reset(level_infra=level_infra, common_infra=common_infra)
        strategy._trade_exchange = exchange

        return strategy

    def test_generates_buy_order(self, strategy):
        self._build_infra(strategy, {"cash": 1_000_000.0, "positions": {}})
        decision = strategy.generate_trade_decision()

        buys = [o for o in decision.get_decision() if o.direction == OrderDir.BUY]
        assert len(buys) == 1
        assert buys[0].stock_id == "SH600000"
        assert buys[0].amount > 0

    def test_generates_sell_when_holding_expired(self, strategy):
        position_state = {
            "cash": 900_000.0,
            "positions": {
                "SH600000": {"amount": 1000, "price": 10.0, "count": 5},  # held >= holding_period
            },
        }
        self._build_infra(strategy, position_state)
        decision = strategy.generate_trade_decision()

        sells = [o for o in decision.get_decision() if o.direction == OrderDir.SELL]
        assert len(sells) == 1
        assert sells[0].stock_id == "SH600000"
        assert sells[0].amount == 1000

    def test_no_buy_when_at_max_positions(self, strategy):
        position_state = {
            "cash": 1_000_000.0,
            "positions": {
                "SH600001": {"amount": 1000, "price": 10.0, "count": 1},
            },
        }
        # Recreate strategy with max_positions=1 to fill capacity.
        # The fixture already builds strategy with max_positions=5, so we build a new one.
        strategy = GoldenTriangleStrategy(
            filter_st=False,
            max_positions=1,
            holding_period=3,
            sell_on_exit_signal=False,
        )
        strategy._data_source.fetch = lambda instruments, start, end: _make_ohlcv_turnover()
        strategy._data_source.fetch_stock_info = lambda: pd.DataFrame()
        self._build_infra(strategy, position_state, trade_step=30, price=10.0)

        decision = strategy.generate_trade_decision()
        assert not any(o.direction == OrderDir.BUY for o in decision.get_decision())
