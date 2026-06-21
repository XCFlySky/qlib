# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
HybridDataSource
================
Fetches OHLCV efficiently through Qlib's binary backend and obtains
turnover (换手率) via AKShare / Tushare / Baostock / local CSV.
"""

import os
import time
from pathlib import Path
from typing import Optional, Union, List, Iterable
import pandas as pd
import numpy as np
from loguru import logger


class TurnoverFetcher:
    """Pluggable turnover data fetcher."""

    @staticmethod
    def from_qlib(instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Try to load ``$turnover`` directly from Qlib data storage."""
        try:
            from qlib.data import D
            df = D.features(instruments, fields=["$turnover"], start_time=start_date, end_time=end_date, freq="day")
            if not df.empty:
                df = df.rename(columns={"$turnover": "turnover"})
            return df
        except Exception as e:
            logger.warning(f"Qlib turnover not available: {e}")
            return pd.DataFrame()

    @staticmethod
    def from_akshare_spot() -> pd.DataFrame:
        """
        Fetch **today's** turnover for the entire A-share market in a single request.
        Returns DataFrame indexed by instrument (Qlib format, e.g. SH600000).
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("akshare is required. Install it via: pip install akshare")

        logger.info("Fetching real-time turnover via akshare.stock_zh_a_spot_em ...")
        df = ak.stock_zh_a_spot_em()
        # Map columns
        df = df.rename(columns={
            "代码": "code",
            "名称": "name",
            "换手率": "turnover",
        })
        df["instrument"] = df["code"].apply(TurnoverFetcher._em_code_to_qlib)
        df = df.dropna(subset=["instrument"])
        df = df.set_index("instrument")
        # turnover is a percentage string like "5.32%" or float 5.32
        if df["turnover"].dtype == object:
            df["turnover"] = df["turnover"].astype(str).str.replace("%", "", regex=False)
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
        return df[["turnover"]]

    @staticmethod
    def from_akshare_hist(
        instruments: Iterable[str],
        start_date: str,
        end_date: str,
        delay: float = 0.5,
    ) -> pd.DataFrame:
        """
        Fetch historical turnover per stock via ``akshare.stock_zh_a_hist``.
        This issues one request per stock and can be slow for 5000+ stocks.
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("akshare is required. Install it via: pip install akshare")

        records = []
        total = len(list(instruments))
        logger.info(f"Fetching historical turnover from akshare for {total} stocks ...")
        for idx, inst in enumerate(instruments, 1):
            symbol = TurnoverFetcher._qlib_code_to_em(inst)
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="",
                )
                if df is None or df.empty:
                    continue
                df = df.rename(columns={
                    "日期": "datetime",
                    "换手率": "turnover",
                })
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["instrument"] = inst
                df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
                records.append(df[["datetime", "instrument", "turnover"]])
            except Exception as e:
                logger.debug(f"akshare hist failed for {inst} ({symbol}): {e}")
            if idx % 100 == 0:
                logger.info(f"... processed {idx}/{total} stocks")
            time.sleep(delay)

        if not records:
            return pd.DataFrame()
        df = pd.concat(records, ignore_index=True)
        df = df.set_index(["datetime", "instrument"]).sort_index()
        return df

    @staticmethod
    def from_tushare(
        instruments: Iterable[str],
        start_date: str,
        end_date: str,
        api_token: Optional[str] = None,
        delay: float = 0.3,
    ) -> pd.DataFrame:
        """
        Fetch turnover via Tushare pro API.

        Uses ``daily_basic(trade_date=...)`` in a date loop, which is much faster
        than one request per stock when backtesting the full market.
        """
        try:
            import tushare as ts
        except ImportError:
            raise ImportError("tushare is required. Install it via: pip install tushare")

        if api_token is None:
            api_token = os.environ.get("TUSHARE_TOKEN")
        if not api_token:
            raise ValueError("Tushare API token is required (pass api_token or set TUSHARE_TOKEN env var)")

        pro = ts.pro_api(api_token)

        # Resolve trading calendar.
        logger.info("Fetching trading calendar from tushare ...")
        cal = pro.trade_cal(
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            is_open="1",
        )
        if cal is None or cal.empty:
            logger.warning("Tushare returned empty trade calendar.")
            return pd.DataFrame()
        trade_dates = sorted(cal["cal_date"].tolist())
        logger.info(f"Total trading days in range: {len(trade_dates)}")

        # Filter set for custom instrument lists.
        inst_filter = None
        if instruments is not None:
            instruments = list(instruments)
            if instruments and instruments != ["all"] and instruments != "all":
                inst_filter = set(instruments)

        records = []
        for idx, td in enumerate(trade_dates, 1):
            try:
                df = pro.daily_basic(trade_date=td)
                if df is None or df.empty:
                    continue
                df = df.rename(columns={
                    "trade_date": "datetime",
                    "turnover_rate": "turnover",
                })
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["instrument"] = df["ts_code"].apply(
                    lambda x: f"SH{x[:6]}" if x.endswith(".SH") else f"SZ{x[:6]}"
                )
                if inst_filter is not None:
                    df = df[df["instrument"].isin(inst_filter)]
                df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
                records.append(df[["datetime", "instrument", "turnover"]])
            except Exception as e:
                logger.debug(f"tushare failed for {td}: {e}")
            if idx % 100 == 0:
                logger.info(f"... tushare processed {idx}/{len(trade_dates)} trade dates")
            time.sleep(delay)

        if not records:
            return pd.DataFrame()
        df = pd.concat(records, ignore_index=True)
        df = df.set_index(["datetime", "instrument"]).sort_index()
        return df

    @staticmethod
    def fetch_st_list() -> List[str]:
        """Fetch current ST stock codes via akshare."""
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("akshare is required. pip install akshare")
        df = ak.stock_info_a_code_name()
        # ST names typically contain "ST" or "*ST"
        st_mask = df["name"].str.contains(r"\*?ST", regex=True, na=False)
        st_codes = df.loc[st_mask, "code"].astype(str).tolist()
        return [TurnoverFetcher._em_code_to_qlib(c) for c in st_codes if TurnoverFetcher._em_code_to_qlib(c)]

    @staticmethod
    def from_csv(path: Union[str, Path]) -> pd.DataFrame:
        """
        Load turnover from a local CSV with columns:
        ``datetime,instrument,turnover`` or ``date,code,turnover``.
        """
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]

        # Normalise datetime
        dt_col = "datetime" if "datetime" in df.columns else "date"
        df[dt_col] = pd.to_datetime(df[dt_col])

        # Normalise instrument
        inst_col = "instrument" if "instrument" in df.columns else "code"
        if inst_col == "code":
            df["instrument"] = df["code"].apply(TurnoverFetcher._em_code_to_qlib)
        df = df.set_index([dt_col, "instrument"]).sort_index()
        return df[["turnover"]]

    # ---- code converters --------------------------------------------
    @staticmethod
    def _qlib_code_to_em(inst: str) -> str:
        """SH600000 -> 600000; SZ000001 -> 000001"""
        return inst[-6:]

    @staticmethod
    def _em_code_to_qlib(code: str) -> Optional[str]:
        """600000 -> SH600000; 000001 -> SZ000001; 300001 -> SZ300001"""
        code = str(code).strip()
        if len(code) != 6 or not code.isdigit():
            return None
        if code.startswith("6"):
            return f"SH{code}"
        elif code.startswith(("0", "3")):
            return f"SZ{code}"
        return None

    @staticmethod
    def _qlib_code_to_ts(inst: str) -> str:
        """SH600000 -> 600000.SH; SZ000001 -> 000001.SZ"""
        return f"{inst[-6:]}.{inst[:2]}"


class HybridDataSource:
    """
    Combines Qlib (fast OHLCV) with an external turnover source.

    Parameters
    ----------
    provider_uri : str, optional
        Qlib data directory.  If None, uses Qlib default.
    region : str, optional
        Qlib region, default ``cn``.
    turnover_source : str
        One of ``qlib``, ``akshare_spot``, ``akshare_hist``, ``tushare``, ``csv``.
    turnover_kwargs : dict
        Extra arguments forwarded to the turnover fetcher
        (e.g. ``{"path": "/data/turnover.csv"}`` or ``{"api_token": "xxx"}``).
    """

    def __init__(
        self,
        provider_uri: Optional[str] = None,
        region: str = "cn",
        turnover_source: str = "akshare_hist",
        turnover_kwargs: Optional[dict] = None,
    ):
        self.provider_uri = provider_uri
        self.region = region
        self.turnover_source = turnover_source
        self.turnover_kwargs = turnover_kwargs or {}
        self._qlib_inited = False

        # Caches for turnover data so that repeated fetch() calls during a
        # backtest do not hit the network every time.
        self._turnover_cache: Optional[pd.DataFrame] = None
        self._turnover_cache_key: Optional[str] = None
        self._turnover_cache_dir = Path.home() / ".qlib" / "cache" / "golden_triangle" / "turnover"

    def _init_qlib(self):
        if self._qlib_inited:
            return
        import qlib
        from qlib.constant import REG_CN
        region_map = {"cn": REG_CN}
        qlib.init(
            provider_uri=self.provider_uri,
            region=region_map.get(self.region, REG_CN),
        )
        self._qlib_inited = True

    def fetch(
        self,
        instruments: Union[str, List[str]],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch merged OHLCV + turnover data.

        Parameters
        ----------
        instruments : str or list of str
            Qlib instrument pool, e.g. ``"csi300"``, ``"all"``, or a list of
            codes like ``["SH600000", "SZ000001"]``.
        start_date : str
            Start date (inclusive), e.g. ``"2024-01-01"``.
        end_date : str
            End date (inclusive).
        fields : list of str, optional
            Additional Qlib fields beyond ``$close`` and ``$volume``.

        Returns
        -------
        pd.DataFrame
            MultiIndex (datetime, instrument) with at least
            ``close``, ``volume``, ``turnover``.
        """
        self._init_qlib()
        from qlib.data import D

        qlib_fields = ["$close", "$volume"]
        if fields:
            qlib_fields += [f if f.startswith("$") else f"${f}" for f in fields]
        qlib_fields = list(dict.fromkeys(qlib_fields))  # preserve order, remove dupes

        logger.info(f"Fetching Qlib fields {qlib_fields} for {instruments} ...")
        df = D.features(instruments, qlib_fields, start_time=start_date, end_time=end_date, freq="day")
        if df.empty:
            raise ValueError("Qlib returned empty DataFrame. Check provider_uri and date range.")

        # Normalise column names
        rename_map = {c: c.replace("$", "").lower() for c in df.columns}
        df = df.rename(columns=rename_map)

        # ---- turnover ----------------------------------------------------
        inst_list = df.index.get_level_values("instrument").unique().tolist()
        df_turnover = self._fetch_turnover(inst_list, start_date, end_date)

        if df_turnover.empty:
            logger.warning("Turnover data is empty – returning OHLCV only.")
            df["turnover"] = np.nan
        else:
            df = df.join(df_turnover, how="left")
            missing_turnover = df["turnover"].isna().sum()
            if missing_turnover:
                logger.warning(f"{missing_turnover} rows have missing turnover.")

        return df

    def _fetch_turnover(self, instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        src = self.turnover_source.lower()
        logger.info(f"Turnover source: {src}")

        if src == "qlib":
            return TurnoverFetcher.from_qlib(instruments, start_date, end_date)

        if src == "akshare_spot":
            # Only returns the latest trading day; useful for end-of-day scans.
            return TurnoverFetcher.from_akshare_spot()

        if src == "akshare_hist":
            return TurnoverFetcher.from_akshare_hist(instruments, start_date, end_date)

        if src == "tushare":
            # Tushare's daily_basic(trade_date=...) returns the whole market for
            # each date.  Cache the full-market result once and filter locally on
            # every subsequent fetch() call inside the backtest loop.
            return self._fetch_turnover_cached(
                instruments, start_date, end_date,
                fetcher=lambda: TurnoverFetcher.from_tushare(
                    None, start_date, end_date, **self.turnover_kwargs
                ),
            )

        if src == "csv":
            path = self.turnover_kwargs.get("path")
            if not path:
                raise ValueError("turnover_kwargs must contain 'path' when turnover_source='csv'")
            return TurnoverFetcher.from_csv(path)

        raise ValueError(f"Unknown turnover_source: {src}")

    def _fetch_turnover_cached(
        self,
        instruments: List[str],
        start_date: str,
        end_date: str,
        fetcher: callable,
    ) -> pd.DataFrame:
        """
        Fetch turnover with both in-memory and on-disk caching.

        The cache key is built from the source type and the date range, so the
        same backtest will reuse previously downloaded data across runs.
        """
        cache_key = f"{self.turnover_source.lower()}_{start_date}_{end_date}"

        # 1. In-memory cache (same backtest, repeated fetch() calls)
        if self._turnover_cache_key == cache_key and self._turnover_cache is not None:
            logger.info("Using in-memory cached turnover data")
            df = self._turnover_cache
        else:
            # 2. On-disk cache (across separate script runs)
            cache_path = self._turnover_cache_dir / f"{cache_key}.feather"
            if cache_path.exists():
                logger.info(f"Loading turnover cache from {cache_path}")
                df = pd.read_feather(cache_path).set_index(["datetime", "instrument"]).sort_index()
            else:
                logger.info("Downloading turnover data from remote ...")
                df = fetcher()
                if not df.empty:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.reset_index().to_feather(cache_path)
                    logger.info(f"Saved turnover cache to {cache_path}")

            self._turnover_cache = df
            self._turnover_cache_key = cache_key

        # Filter to the instruments requested by the current backtest step.
        if instruments is not None and not df.empty:
            inst_filter = set(instruments)
            df = df[df.index.get_level_values("instrument").isin(inst_filter)]
        return df

    def fetch_stock_info(self) -> pd.DataFrame:
        """
        Fetch basic stock info (name) via akshare.
        Returns DataFrame indexed by instrument.
        Note: industry data is not available in stock_info_a_code_name.
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("akshare is required for fetch_stock_info")

        logger.info("Fetching stock info via akshare ...")
        df = ak.stock_info_a_code_name()
        df = df.rename(columns={
            "code": "code",
            "name": "name",
        })
        df["instrument"] = df["code"].apply(TurnoverFetcher._em_code_to_qlib)
        df = df.dropna(subset=["instrument"])
        df = df.set_index("instrument")
        return df[["name"]]
