"""
Robust cross-market data helper (drop-in replacement for get_close / get_close_multi).

Problem fixed
-------------
The original code used akshare's East Money endpoints:
    ak.stock_hk_hist  (HK)
    ak.stock_zh_a_hist / ak.stock_zh_index_daily_em  (A-share / index)

In this network environment the East Money server (33.push2his.eastmoney.com)
resets the TCP connection (ConnectionError / RemoteDisconnected).

This version switches to Sina-based akshare endpoints that DO work here:
    ak.stock_hk_daily      -> HK
    ak.stock_zh_a_daily    -> A-share
    ak.stock_zh_index_daily-> indices
    ak.stock_us_daily      -> US (already Sina-based)

It also adds retry/backoff, inter-request delays, and optional fallbacks
(Tushare Pro for A-share/HK, yfinance for HK/US) if the Sina source fails.
"""

from __future__ import annotations

import os
import re
import time
import random
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional fallback backends
# ---------------------------------------------------------------------------
try:
    import akshare as ak

    _HAS_AK = True
except Exception as e:
    _HAS_AK = False
    ak = None

try:
    import yfinance as yf
    import requests

    _YF_SESSION = requests.Session()
    _YF_SESSION.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
    )
    _HAS_YF = True
except Exception:
    _HAS_YF = False
    _YF_SESSION = None

try:
    import tushare as ts

    _HAS_TS = True
except Exception:
    _HAS_TS = False

# ---------------------------------------------------------------------------
# Tushare token (only needed if you use the Tushare fallback path)
# ---------------------------------------------------------------------------
TUSHARE_TOKEN: Optional[str] = None


def _get_pro():
    if not _HAS_TS:
        raise ImportError("tushare 未安装")
    token = TUSHARE_TOKEN or os.environ.get("TUSHARE_TOKEN")
    if not token or token.strip() in ("", "YOUR_TUSHARE_TOKEN_HERE"):
        raise ValueError(
            "请设置 Tushare Pro token：修改 TUSHARE_TOKEN 变量或设置环境变量 TUSHARE_TOKEN"
        )
    ts.set_token(token)
    return ts.pro_api()


# ---------------------------------------------------------------------------
# Ticker classification / period parsing
# ---------------------------------------------------------------------------
_YF_SPECIAL = {
    "^GSPC": ".INX",
    "^DJI": ".DJI",
    "^IXIC": ".IXIC",
    "GC=F": "GC",
    "SI=F": "SI",
    "CL=F": "CL",
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
}

_A_INDEX_CODES = {"000300", "000016", "000905", "000852", "000001"}


def _parse_period(period: str):
    """Return (start_date, end_date) as 'YYYYMMDD' strings."""
    end = datetime.today()
    m = re.match(r"^(\d+)\s*(y|mo|w|d)$", period.lower().strip())
    if not m:
        days = 365 * 3
    else:
        days = int(m.group(1)) * {"y": 365, "mo": 30, "w": 7, "d": 1}[m.group(2)]
    # 30-day buffer so boundary dates are not lost
    return (end - timedelta(days=days + 30)).strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _classify(ticker: str):
    """Return (kind, sina_symbol, raw_ticker)."""
    t = ticker.strip()
    if t in _YF_SPECIAL:
        return ("yf", _YF_SPECIAL[t], t)
    if t.endswith((".SS", ".SH", ".SZ")):
        code = t.split(".")[0]
        market = "sh" if t.endswith((".SS", ".SH")) else "sz"
        if code in _A_INDEX_CODES:
            return ("index", f"{market}{code}", code)
        return ("a", f"{market}{code}", code)
    if t.endswith(".HK"):
        code = t.split(".")[0].zfill(5)
        return ("hk", code, t)
    return ("us", t, t)


def _norm(df, date_col: str, close_col: str) -> pd.Series:
    """Normalize DataFrame to datetime-indexed Series named 'Close'."""
    out = df[[date_col, close_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.set_index(date_col).sort_index()[close_col].astype(float).rename("Close")


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def _retry_call(func, max_retry: int = 3, sleep_base: float = 1.0, label: str = ""):
    """Retry on ConnectionError / TimeoutError / OSError with exponential backoff."""
    last_err = None
    for attempt in range(max_retry):
        try:
            return func()
        except (ConnectionError, TimeoutError, OSError) as e:
            last_err = e
            if attempt == max_retry - 1:
                break
            sleep_sec = sleep_base * (2 ** attempt) + random.uniform(0.5, 1.5)
            print(f"  [{label}] 连接失败: {e} | 第 {attempt + 1} 次重试，等待 {sleep_sec:.1f} 秒...")
            time.sleep(sleep_sec)
    raise last_err


# ---------------------------------------------------------------------------
# Data fetchers (Sina-based akshare)
# ---------------------------------------------------------------------------
def _ak_a_close(symbol: str, start_dt, end_dt):
    df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
    s = _norm(df, "date", "close")
    return s.loc[start_dt:end_dt]


def _ak_index_close(symbol: str, start_dt, end_dt):
    df = ak.stock_zh_index_daily(symbol=symbol)
    s = _norm(df, "date", "close")
    return s.loc[start_dt:end_dt]


def _ak_hk_close(symbol: str, start_dt, end_dt):
    df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
    s = _norm(df, "date", "close")
    return s.loc[start_dt:end_dt]


def _ak_us_close(symbol: str, start_dt, end_dt):
    df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
    s = _norm(df, "date", "close")
    return s.loc[start_dt:end_dt]


# ---------------------------------------------------------------------------
# Fallback fetchers
# ---------------------------------------------------------------------------
def _yf_download_safe(ticker: str, period: str, max_retry: int = 5):
    """yfinance download with long random delays and exponential backoff."""
    if not _HAS_YF:
        raise ImportError("yfinance 未安装")
    for attempt in range(max_retry):
        try:
            if attempt > 0:
                sleep_sec = 2 ** attempt + random.uniform(2, 5)
                print(f"  [yfinance] {ticker} 第 {attempt + 1} 次尝试，等待 {sleep_sec:.1f} 秒...")
                time.sleep(sleep_sec)
            else:
                time.sleep(random.uniform(3, 6))

            s = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                session=_YF_SESSION,
            )["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            if s is None or s.empty:
                raise ValueError("empty")
            return s.rename("Close")
        except Exception as e:
            err_msg = str(e)
            if "Rate limited" in err_msg or "Too Many Requests" in err_msg:
                if attempt == max_retry - 1:
                    raise ValueError(
                        f"{ticker} 被 Yahoo Finance 限流，请稍后重试或加代理/VPN"
                    )
                continue
            if attempt < max_retry - 1:
                time.sleep(random.uniform(3, 6))
                continue
            raise
    raise ValueError(f"{ticker} yfinance 下载失败")


def _ts_a_close(ts_code: str, start: str, end: str):
    pro = _get_pro()
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        raise ValueError("empty")
    return _norm(df, "trade_date", "close")


def _ts_hk_close(ts_code: str, start: str, end: str):
    pro = _get_pro()
    df = pro.hk_daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        raise ValueError("empty")
    return _norm(df, "trade_date", "close")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_close(ticker: str, period: str = "3y"):
    """Fetch adjusted close prices for one ticker.

    Ticker examples:
        A-share : 600519.SS, 000001.SZ
        HK      : 0700.HK
        US      : AAPL, TSLA
        Index   : 000300.SS
    """
    if not _HAS_AK:
        raise ImportError("akshare 未安装，请先 pip install akshare")

    start, end = _parse_period(period)
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    kind, symbol, raw = _classify(ticker)

    try:
        if kind == "yf":
            return _yf_download_safe(symbol, period)
        if kind == "a":
            return _retry_call(
                lambda: _ak_a_close(symbol, start_dt, end_dt),
                max_retry=3,
                label=f"akshare-A-{ticker}",
            )
        if kind == "index":
            return _retry_call(
                lambda: _ak_index_close(symbol, start_dt, end_dt),
                max_retry=3,
                label=f"akshare-idx-{ticker}",
            )
        if kind == "hk":
            return _retry_call(
                lambda: _ak_hk_close(symbol, start_dt, end_dt),
                max_retry=3,
                label=f"akshare-HK-{ticker}",
            )
        if kind == "us":
            return _retry_call(
                lambda: _ak_us_close(symbol, start_dt, end_dt),
                max_retry=3,
                label=f"akshare-US-{ticker}",
            )
    except Exception as e:
        print(f"  [警告] akshare {ticker} 主源失败: {e}")

        # Fallbacks
        if kind in ("a",) and _HAS_TS:
            print(f"  [回退] 尝试 Tushare {raw} ...")
            market = "SH" if raw.endswith((".SS", ".SH")) else "SZ"
            ts_code = f"{raw.split('.')[0]}.{market}"
            return _ts_a_close(ts_code, start, end)

        if kind == "index" and _HAS_TS:
            print(f"  [回退] 尝试 Tushare {raw} ...")
            market = "SH" if raw.endswith((".SS", ".SH")) else "SZ"
            ts_code = f"{raw.split('.')[0]}.{market}"
            return _ts_a_close(ts_code, start, end)

        if kind == "hk" and _HAS_TS:
            print(f"  [回退] 尝试 Tushare {raw} ...")
            return _ts_hk_close(symbol + ".HK", start, end)

        if kind in ("hk", "us", "yf") and _HAS_YF:
            print(f"  [回退] 尝试 yfinance {raw} ...")
            return _yf_download_safe(raw, period)

        raise

    raise ValueError(f"不支持的 ticker 类型: {ticker}")


def get_close_multi(tickers: Dict[str, str], period: str = "3y"):
    """Batch fetch close prices.  Returns a DataFrame aligned by trading dates."""
    series: Dict[str, pd.Series] = {}
    for i, (name, t) in enumerate(tickers.items()):
        try:
            if i > 0:
                time.sleep(random.uniform(1.0, 2.5))
            series[name] = get_close(t, period=period)
            print(f"[OK] {name} ({t}) 获取成功")
        except Exception as e:
            print(f"[SKIP] {name} ({t}): {e}")

    if not series:
        raise ValueError("所有标的获取失败")

    df = pd.concat(series, axis=1).sort_index()
    return df


if __name__ == "__main__":
    tickers = {"茅台": "600519.SS", "腾讯": "0700.HK", "苹果": "AAPL"}
    df = get_close_multi(tickers, period="5y")
    print(df.head())
    print(df.tail())
    print("shape:", df.shape)
