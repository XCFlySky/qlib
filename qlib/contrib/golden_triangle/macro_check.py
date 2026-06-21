# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Macro Market Check (全球股市宏观检查)
========================================

Checks overnight global market performance to provide a risk
backdrop for A-share stock selection.

Monitored indices:
- US: 道琼斯 (DJI), 纳斯达克 (IXIC), 标普500 (SPX)
- Asia: 日经225 (N225), 韩国KOSPI (KS11)

Usage::

    from qlib.contrib.golden_triangle.macro_check import MacroCheck
    macro = MacroCheck()
    report = macro.check(trade_date="2024-09-24")
    print(report)

"""

from typing import Optional, Dict
import pandas as pd
from loguru import logger


class MacroCheck:
    """
    全球股市情绪检查器

    Parameters
    ----------
    us_threshold : float
        If overnight US major indices drop more than this percentage,
        a caution flag is raised (default 2.0).
    asia_threshold : float
        If Asia indices drop more than this percentage, caution flag
        (default 2.0).
    tushare_token : str, optional
        Tushare Pro API token.  If None, reads ``TUSHARE_TOKEN`` env var.
    """

    INDEX_MAP = {
        "DJI": "道琼斯",      # Dow Jones
        "IXIC": "纳斯达克",   # Nasdaq
        "SPX": "标普500",     # S&P 500
        "N225": "日经225",    # Nikkei 225
        "KS11": "韩国KOSPI",  # KOSPI
    }

    def __init__(
        self,
        us_threshold: float = 2.0,
        asia_threshold: float = 2.0,
        tushare_token: Optional[str] = None,
    ):
        self.us_threshold = us_threshold
        self.asia_threshold = asia_threshold
        self.tushare_token = tushare_token
        self._pro = None

    def _get_pro(self):
        if self._pro is not None:
            return self._pro
        try:
            import tushare as ts
        except ImportError:
            raise ImportError("tushare is required for MacroCheck. pip install tushare")

        token = self.tushare_token or __import__("os").environ.get("TUSHARE_TOKEN")
        if not token:
            raise ValueError("Tushare API token required. Pass tushare_token or set TUSHARE_TOKEN env var.")
        self._pro = ts.pro_api(token)
        return self._pro

    def fetch_index_change(self, ts_code: str, trade_date: str) -> Optional[float]:
        """
        Fetch the previous trading day's pct change for a given index.
        Returns pct_chg (e.g. -1.5 means -1.5%%) or None on failure.
        """
        pro = self._get_pro()
        try:
            df = pro.index_global(ts_code=ts_code, trade_date=trade_date)
            if df is None or df.empty:
                return None
            return float(df.iloc[0]["pct_chg"])
        except Exception as e:
            logger.debug(f"MacroCheck failed for {ts_code}@{trade_date}: {e}")
            return None

    def check(self, trade_date: Optional[str] = None) -> Dict[str, object]:
        """
        Run macro check for the trading day *prior* to ``trade_date``.

        Parameters
        ----------
        trade_date : str, optional
            The A-share anchor date (YYYY-MM-DD).  If None, uses today.

        Returns
        -------
        dict
            {
                "us": {"DJI": -0.5, "IXIC": -1.2, ...},
                "asia": {"N225": 0.8, "KS11": -0.3, ...},
                "caution": True/False,
                "reason": "...",
            }
        """
        if trade_date is None:
            trade_date = pd.Timestamp.now().strftime("%Y%m%d")
        else:
            trade_date = pd.Timestamp(trade_date).strftime("%Y%m%d")

        report = {
            "us": {},
            "asia": {},
            "caution": False,
            "reason": "",
        }

        reasons = []

        # US markets (previous close relative to trade_date)
        for code, name in self.INDEX_MAP.items():
            if code in ("N225", "KS11"):
                continue
            chg = self.fetch_index_change(code, trade_date)
            if chg is not None:
                report["us"][name] = round(chg, 2)
                if chg <= -self.us_threshold:
                    reasons.append(f"美股{name}大跌{chg:.2f}%")

        # Asia markets
        for code, name in self.INDEX_MAP.items():
            if code not in ("N225", "KS11"):
                continue
            chg = self.fetch_index_change(code, trade_date)
            if chg is not None:
                report["asia"][name] = round(chg, 2)
                if chg <= -self.asia_threshold:
                    reasons.append(f"亚太{name}大跌{chg:.2f}%")

        if reasons:
            report["caution"] = True
            report["reason"] = "；".join(reasons)
        else:
            report["reason"] = "隔夜外围环境平稳或偏暖"

        return report

    def print_report(self, trade_date: Optional[str] = None):
        """Pretty-print macro report."""
        r = self.check(trade_date)
        print("=" * 50)
        print("全球股市宏观检查")
        print("=" * 50)
        print(f"美股：{r['us']}")
        print(f"亚太：{r['asia']}")
        flag = "⚠️  caution" if r["caution"] else "✅  normal"
        print(f"情绪：{flag}")
        print(f"原因：{r['reason']}")
        print("=" * 50)
        return r
