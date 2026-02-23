"""
Strategy Layer : génération des signaux de trading.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI = 100 - 100/(1 + RS) avec RS = moyenne gains / moyenne pertes"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    return 100 - (100 / (1 + rs))


def generate_signals(
    df: pd.DataFrame,
    sma_fast: int = 20,
    sma_slow: int = 50,
    short_allowed: bool = False,
    use_rsi_filter: bool = False,
    rsi_period: int = 14,
    rsi_long_max: float = 70.0,
    rsi_short_min: float = 30.0,
    use_volume_filter: bool = False,
    volume_ma_period: int = 20,
    start_in_cash: bool = True,
) -> pd.DataFrame:
    """
    SMA crossover : SMA_fast > SMA_slow → 1 (Achat)
                   SMA_fast < SMA_slow → -1 (Vente à découvert) ou 0 (Sortie)
    """
    df = df.copy()
    df["SMA_fast"] = df["Close"].rolling(sma_fast).mean()
    df["SMA_slow"] = df["Close"].rolling(sma_slow).mean()

    if short_allowed:
        df["signal_raw"] = np.where(df["SMA_fast"] > df["SMA_slow"], 1, -1)
    else:
        df["signal_raw"] = np.where(df["SMA_fast"] > df["SMA_slow"], 1, 0)

    if use_rsi_filter:
        df["RSI"] = compute_rsi(df["Close"], rsi_period)
        long_ok = df["RSI"] < rsi_long_max
        short_ok = df["RSI"] > rsi_short_min
        df["signal_raw"] = np.where(
            df["signal_raw"] == 1,
            np.where(long_ok, 1, 0),
            np.where(df["signal_raw"] == -1, np.where(short_ok, -1, 0), 0),
        )

    if use_volume_filter:
        df["volume_ma"] = df["Volume"].rolling(volume_ma_period).mean()
        vol_ok = df["Volume"] >= df["volume_ma"]
        df["signal_raw"] = np.where(vol_ok, df["signal_raw"], 0)

    df["signal"] = df["signal_raw"].shift(1)
    df["signal"] = df["signal"].fillna(0).astype(int)

    if start_in_cash:
        warmup = min(sma_slow, len(df))
        df.loc[df.index[:warmup], "signal"] = 0

    return df


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands : middle = SMA, upper = middle + k*std, lower = middle - k*std"""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def generate_signals_bollinger(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    use_volume_filter: bool = True,
    volume_ma_period: int = 20,
    short_allowed: bool = False,
    start_in_cash: bool = True,
) -> pd.DataFrame:
    """Mean Reversion : Bollinger Bands + RSI + Volume."""
    df = df.copy()
    upper, middle, lower = compute_bollinger_bands(df["Close"], bb_period, bb_std)
    df["BB_upper"] = upper
    df["BB_middle"] = middle
    df["BB_lower"] = lower
    df["RSI"] = compute_rsi(df["Close"], rsi_period)

    touch_lower = df["Low"] <= df["BB_lower"]
    touch_upper = df["High"] >= df["BB_upper"]
    touch_middle = df["Close"] >= df["BB_middle"]

    entry_long = touch_lower & (df["RSI"] < rsi_oversold)
    exit_long = touch_upper | touch_middle

    if use_volume_filter:
        vol_ma = df["Volume"].rolling(volume_ma_period).mean()
        entry_long = entry_long & (df["Volume"] >= vol_ma)

    entry_long = entry_long.shift(1).fillna(False)

    signal = np.zeros(len(df), dtype=int)
    position = 0
    for i in range(bb_period if start_in_cash else 0, len(df)):
        if position == 0:
            if entry_long.iloc[i]:
                position = 1
            elif short_allowed and touch_upper.iloc[i] and df["RSI"].iloc[i] > rsi_overbought:
                position = -1
        elif position == 1:
            if exit_long.iloc[i]:
                position = 0
        elif position == -1:
            if touch_lower.iloc[i] or (df["Close"].iloc[i] <= df["BB_middle"].iloc[i]):
                position = 0
        signal[i] = position

    df["signal_raw"] = signal
    df["signal"] = pd.Series(signal, index=df.index).fillna(0).astype(int)
    return df


def generate_signals_buy_hold(df: pd.DataFrame) -> pd.DataFrame:
    """Buy & Hold : toujours en position long."""
    df = df.copy()
    df["signal"] = 1
    df["signal_raw"] = 1
    return df


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD = EMA(fast) - EMA(slow), Signal = EMA(MACD, signal_period), Histogram = MACD - Signal"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def generate_signals_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    start_in_cash: bool = True,
) -> pd.DataFrame:
    """MACD Crossover : achat quand MACD croise au-dessus de la ligne de signal."""
    df = df.copy()
    macd_line, signal_line, histogram = compute_macd(df["Close"], fast, slow, signal_period)
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = histogram

    df["signal_raw"] = np.where(macd_line > signal_line, 1, 0)
    df["signal"] = df["signal_raw"].shift(1).fillna(0).astype(int)

    warmup = slow + signal_period if start_in_cash else 0
    if start_in_cash and warmup > 0:
        df.loc[df.index[:warmup], "signal"] = 0
    return df


def generate_signals_rsi_naive(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    start_in_cash: bool = True,
) -> pd.DataFrame:
    """RSI seul (naïf) : achète RSI < 30, vend RSI > 70."""
    df = df.copy()
    df["RSI"] = compute_rsi(df["Close"], rsi_period)
    df["signal_raw"] = np.where(df["RSI"] < rsi_oversold, 1, np.where(df["RSI"] > rsi_overbought, 0, np.nan))
    df["signal_raw"] = df["signal_raw"].ffill().fillna(0).astype(int)
    df["signal"] = df["signal_raw"].shift(1).fillna(0).astype(int)

    if start_in_cash:
        df.loc[df.index[:rsi_period + 1], "signal"] = 0
    return df


def generate_signals_inverse_sma(
    df: pd.DataFrame,
    sma_fast: int = 20,
    sma_slow: int = 50,
    start_in_cash: bool = True,
) -> pd.DataFrame:
    """Inverse SMA : achète quand SMA rapide < SMA lente (contrarian)."""
    df = df.copy()
    df["SMA_fast"] = df["Close"].rolling(sma_fast).mean()
    df["SMA_slow"] = df["Close"].rolling(sma_slow).mean()
    df["signal_raw"] = np.where(df["SMA_fast"] < df["SMA_slow"], 1, 0)
    df["signal"] = df["signal_raw"].shift(1).fillna(0).astype(int)

    if start_in_cash:
        df.loc[df.index[:sma_slow], "signal"] = 0
    return df
