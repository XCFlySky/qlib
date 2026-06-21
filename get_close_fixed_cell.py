"""
This is a notebook-cell-sized drop-in replacement for the broken get_close/get_close_multi.

Copy the code below (everything after the dashed line) into a Jupyter cell and run it
instead of the old akshare-EastMoney cell.

What changed:
- Replaced EastMoney endpoints (stock_hk_hist, stock_zh_a_hist, stock_zh_index_daily_em)
  with Sina endpoints (stock_hk_daily, stock_zh_a_daily, stock_zh_index_daily)
  that do NOT raise ConnectionError in this network.
- Added retry/backoff for transient network failures.
- Added small delays between batch requests.
"""

# ---------------------------------------------------------------------------
# COPY EVERYTHING BELOW THIS LINE INTO A JUPYTER CELL
# ---------------------------------------------------------------------------

import re
import time
import random
import warnings
from datetime import datetime, timedelta
import pandas as pd
import akshare as ak

warnings.filterwarnings("ignore")


def _parse_period(period="3y"):
    end = datetime.today()
    m = re.match(r"^(\d+)\s*(y|mo|w|d)$", period.lower().strip())
    days = 365 * 3 if not m else int(m.group(1)) * {"y": 365, "mo": 30, "w": 7, "d": 1}[m.group(2)]
    return (end - timedelta(days=days + 30)).strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _classify(ticker):
    t = ticker.strip()
    if t.endswith((".SS", ".SH", ".SZ")):
        code = t.split(".")[0]
        market = "sh" if t.endswith((".SS", ".SH")) else "sz"
        if code in {"000300", "000016", "000905", "000852", "000001"}:
            return ("index", f"{market}{code}")
        return ("a", f"{market}{code}")
    if t.endswith(".HK"):
        return ("hk", t.split(".")[0].zfill(5))
    return ("us", t)


def _norm(df, date_col, close_col):
    out = df[[date_col, close_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.set_index(date_col).sort_index()[close_col].astype(float).rename("Close")


def _retry(func, max_retry=3, sleep_base=1.0, label=""):
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


def get_close(ticker, period="3y"):
    start, end = _parse_period(period)
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    kind, symbol = _classify(ticker)

    def fetch():
        if kind == "a":
            df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
        elif kind == "index":
            df = ak.stock_zh_index_daily(symbol=symbol)
        elif kind == "hk":
            df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
        else:  # us
            df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
        s = _norm(df, "date", "close")
        return s.loc[start_dt:end_dt]

    return _retry(fetch, max_retry=3, label=f"akshare-{kind}-{ticker}")


def get_close_multi(tickers, period="3y"):
    series = {}
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
    return pd.concat(series, axis=1).sort_index()


print("[OK] get_close / get_close_multi (Sina-akshare 修复版) 已加载")
