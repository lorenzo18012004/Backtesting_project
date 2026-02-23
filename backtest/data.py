"""
Data Pipeline : récupération et nettoyage des données OHLCV.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional

def fetch_ohlcv_yahoo(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 1000,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list:
    """
    Récupère les données OHLCV via Yahoo Finance (actions, ETFs).
    Retourne [timestamp_ms, O, H, L, C, V].
    Timeframes supportés : 1d, 1wk, 1mo (Yahoo limite 1h à ~7 jours).
    """
    interval_map = {"1d": "1d", "1wk": "1wk", "1w": "1wk", "1mo": "1mo", "1m": "1mo"}
    interval = interval_map.get(timeframe, "1d")
    start = pd.Timestamp(since) if since else pd.Timestamp.now() - pd.Timedelta(days=365 * 2)
    end = pd.Timestamp(until) if until else pd.Timestamp.now()
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
    if df.empty or len(df) == 0:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.ffill().bfill()
    df["Volume"] = df["Volume"].fillna(0)
    ts_ms = (df.index.astype(np.int64) // 1_000_000).astype(int)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    vals = df[cols].astype(float).values
    ohlcv = [[int(ts_ms[i])] + vals[i].tolist() for i in range(len(df))]
    if limit and len(ohlcv) > limit and (since is None or until is None):
        ohlcv = ohlcv[-limit:]
    return ohlcv


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int = 1000,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    data_source: str = "yahoo",
) -> list:
    """Récupère les données OHLCV via Yahoo Finance (actions, ETFs)."""
    return fetch_ohlcv_yahoo(symbol, timeframe, limit, since, until)


def clean_ohlcv(ohlcv_raw: list) -> pd.DataFrame:
    """
    Transforme les données brutes en DataFrame propre.
    Colonnes : Open, High, Low, Close, Volume
    Index : DatetimeIndex (ISO8601)
    """
    if not ohlcv_raw or len(ohlcv_raw) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(
        ohlcv_raw,
        columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df.index = df.index.tz_localize(None)
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les rendements logarithmiques : r_t = ln(P_t / P_{t-1})"""
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df
