# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Golden Triangle Stock Selector (有效买托策略)

A multi-factor stock selection model based on Qlib that screens stocks
with "effective buying support" signals using:
1. Moving Average Golden Cross Triangle (MA5 > MA10 > MA20 with recent crosses)
2. Volume surge (>= 1.5x 7-day average)
3. Turnover rate (> 3%)
"""

from .selector import GoldenTriangleSelector
from .data_source import HybridDataSource
from .strategy import GoldenTriangleStrategy

__all__ = ["GoldenTriangleSelector", "HybridDataSource", "GoldenTriangleStrategy"]
