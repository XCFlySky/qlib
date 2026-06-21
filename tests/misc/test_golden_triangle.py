# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unit tests for Golden Triangle (有效买托) stock selector.
"""

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.golden_triangle.selector import GoldenTriangleSelector


class TestGoldenTriangleSelector:
    @staticmethod
    def _make_df(close, volume, turnover, start_date="2024-01-01"):
        dates = pd.date_range(start_date, periods=len(close), freq="B")
        return pd.DataFrame(
            {
                "datetime": dates,
                "instrument": ["SH600000"] * len(close),
                "close": close,
                "volume": volume,
                "turnover": turnover,
            }
        ).set_index(["datetime", "instrument"])

    def test_golden_triangle_detected(self):
        """Basic case: one stock with perfect golden triangle on day 27."""
        close = np.ones(30) * 10.0
        close[27] = 11.0
        close[28] = 15.0
        close[29] = 20.0

        volume = np.ones(30) * 10000
        volume[27] = 30000  # 3x prior average

        turnover = np.ones(30) * 2.0
        turnover[27] = 5.0  # > 3%

        df = self._make_df(close, volume, turnover)
        selector = GoldenTriangleSelector(lookback_days=30, observation_window=3)
        result = selector.select(df, trade_date=df.index.get_level_values("datetime").max())

        assert len(result) == 1
        row = result.iloc[0]
        assert row["instrument"] == "SH600000"
        assert row["volume_ratio"] == pytest.approx(3.0)
        assert row["turnover"] == pytest.approx(5.0)
        # MA values should match manual calculation
        assert row["ma5"] == pytest.approx(10.2, abs=1e-4)
        assert row["ma10"] == pytest.approx(10.1, abs=1e-4)
        assert row["ma20"] == pytest.approx(10.05, abs=1e-4)

    def test_no_cross_no_result(self):
        """Flat price: no cross, expect empty result."""
        close = np.ones(30) * 10.0
        volume = np.ones(30) * 10000
        turnover = np.ones(30) * 5.0

        df = self._make_df(close, volume, turnover)
        selector = GoldenTriangleSelector()
        result = selector.select(df)
        assert result.empty

    def test_volume_filter_rejects(self):
        """Golden triangle present but volume insufficient."""
        close = np.ones(30) * 10.0
        close[27] = 11.0
        close[28] = 15.0
        close[29] = 20.0

        volume = np.ones(30) * 10000
        volume[27] = 12000  # only 1.2x, below 1.5 threshold

        turnover = np.ones(30) * 5.0

        df = self._make_df(close, volume, turnover)
        selector = GoldenTriangleSelector(volume_multiplier=1.5)
        result = selector.select(df)
        assert result.empty

    def test_turnover_filter_rejects(self):
        """Golden triangle + volume OK but turnover too low."""
        close = np.ones(30) * 10.0
        close[27] = 11.0
        close[28] = 15.0
        close[29] = 20.0

        volume = np.ones(30) * 10000
        volume[27] = 30000

        turnover = np.ones(30) * 2.0  # below 3%

        df = self._make_df(close, volume, turnover)
        selector = GoldenTriangleSelector(turnover_threshold=3.0)
        result = selector.select(df)
        assert result.empty

    def test_st_exclusion(self):
        """ST stocks should be excluded when st_set is provided."""
        close = np.ones(30) * 10.0
        close[27] = 11.0
        close[28] = 15.0
        close[29] = 20.0

        volume = np.ones(30) * 10000
        volume[27] = 30000

        turnover = np.ones(30) * 5.0

        df = self._make_df(close, volume, turnover)
        selector = GoldenTriangleSelector()
        result = selector.select(df, st_set={"SH600000"})
        assert result.empty

    def test_golden_triangle_on_last_day(self):
        """GT happens exactly on the anchor date T."""
        close = np.ones(30) * 10.0
        # Trigger GT on the very last day (idx 29)
        close[29] = 11.0

        volume = np.ones(30) * 10000
        volume[29] = 30000

        turnover = np.ones(30) * 2.0
        turnover[29] = 5.0

        df = self._make_df(close, volume, turnover)
        selector = GoldenTriangleSelector(observation_window=3)
        result = selector.select(df, trade_date=df.index.get_level_values("datetime").max())

        assert len(result) == 1
        # cross_date should be the anchor date itself
        assert result.iloc[0]["cross_date"] == df.index.get_level_values("datetime").unique()[-1]

    def test_min_listing_days_filter(self):
        """Newly listed stocks with < 20 days data should be dropped."""
        close = np.ones(15) * 10.0
        close[12] = 11.0
        close[13] = 15.0
        close[14] = 20.0

        volume = np.ones(15) * 10000
        volume[12] = 30000

        turnover = np.ones(15) * 5.0

        df = self._make_df(close, volume, turnover)
        selector = GoldenTriangleSelector(min_listing_days=20)
        result = selector.select(df)
        assert result.empty

    def test_multi_instrument(self):
        """Two instruments: one passes, one fails."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        np.random.seed(0)

        records = []
        for inst in ["SH600000", "SZ000001"]:
            close = np.ones(30) * 10.0
            volume = np.ones(30) * 10000
            turnover = np.ones(30) * 2.0

            if inst == "SH600000":
                close[27] = 11.0
                close[28] = 15.0
                close[29] = 20.0
                volume[27] = 30000
                turnover[27] = 5.0
            else:
                close = close + np.random.randn(30) * 0.1

            for i, d in enumerate(dates):
                records.append(
                    {
                        "datetime": d,
                        "instrument": inst,
                        "close": close[i],
                        "volume": volume[i],
                        "turnover": turnover[i],
                    }
                )

        df = pd.DataFrame(records).set_index(["datetime", "instrument"])
        selector = GoldenTriangleSelector()
        result = selector.select(df)

        assert len(result) == 1
        assert result.iloc[0]["instrument"] == "SH600000"


class TestDataSourceCodecs:
    def test_em_to_qlib(self):
        from qlib.contrib.golden_triangle.data_source import TurnoverFetcher

        assert TurnoverFetcher._em_code_to_qlib("600000") == "SH600000"
        assert TurnoverFetcher._em_code_to_qlib("000001") == "SZ000001"
        assert TurnoverFetcher._em_code_to_qlib("300001") == "SZ300001"
        assert TurnoverFetcher._em_code_to_qlib("invalid") is None

    def test_qlib_to_em(self):
        from qlib.contrib.golden_triangle.data_source import TurnoverFetcher

        assert TurnoverFetcher._qlib_code_to_em("SH600000") == "600000"
        assert TurnoverFetcher._qlib_code_to_em("SZ000001") == "000001"
