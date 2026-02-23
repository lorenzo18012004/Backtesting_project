"""
Backtesting Engine - Système complet de backtesting crypto
Étapes 1 à 7 : Data Pipeline → Data Cleaning → Strategy → PnL → Reality Check → Risk Report → Visualizer
"""

import sys
import io

# Fix encoding Windows pour les accents (sauf sous Streamlit qui gère sa propre sortie)
if sys.platform == "win32" and "streamlit" not in sys.modules:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import Tuple, Optional, List

# Mapping timeframe → durée en secondes
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

# =============================================================================
# ÉTAPE 1 : DATA PIPELINE (Yahoo Finance - actions, ETFs)
# =============================================================================

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
    ohlcv = []
    for ts, row in df.iterrows():
        ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
        ohlcv.append([ts_ms, float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Volume"])])
    # Ne pas tronquer si since/until sont fournis : l'utilisateur a choisi la période
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
    """
    Récupère les données OHLCV via Yahoo Finance (actions, ETFs).
    """
    return fetch_ohlcv_yahoo(symbol, timeframe, limit, since, until)


# =============================================================================
# ÉTAPE 2 : DATA CLEANING (Transformation Pandas)
# =============================================================================

def clean_ohlcv(ohlcv_raw: list) -> pd.DataFrame:
    """
    Transforme les données brutes en DataFrame propre.
    Colonnes : Open, High, Low, Close, Volume
    Index : DatetimeIndex (ISO8601)
    """
    df = pd.DataFrame(
        ohlcv_raw,
        columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    # Conversion timestamp (ms) → datetime ISO8601
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df.index = df.index.tz_localize(None)  # Optionnel : retirer timezone pour affichage
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements logarithmiques : r_t = ln(P_t / P_{t-1})
    """
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


# =============================================================================
# ÉTAPE 3 : STRATEGY LAYER (Génération de signaux)
# =============================================================================

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
    Filtres optionnels : RSI (éviter overbought/oversold), Volume (confirmer la tendance)
    Shift(1) anti-triche : on utilise l'info d'hier pour trader aujourd'hui.
    """
    df = df.copy()
    df["SMA_fast"] = df["Close"].rolling(sma_fast).mean()
    df["SMA_slow"] = df["Close"].rolling(sma_slow).mean()

    # Signal brut : 1 = long, -1 = short, 0 = cash
    if short_allowed:
        df["signal_raw"] = np.where(df["SMA_fast"] > df["SMA_slow"], 1, -1)
    else:
        df["signal_raw"] = np.where(df["SMA_fast"] > df["SMA_slow"], 1, 0)

    # Filtre RSI : ne pas acheter si RSI > rsi_long_max, ne pas short si RSI < rsi_short_min
    if use_rsi_filter:
        df["RSI"] = compute_rsi(df["Close"], rsi_period)
        long_ok = df["RSI"] < rsi_long_max
        short_ok = df["RSI"] > rsi_short_min
        df["signal_raw"] = np.where(
            df["signal_raw"] == 1,
            np.where(long_ok, 1, 0),
            np.where(df["signal_raw"] == -1, np.where(short_ok, -1, 0), 0),
        )

    # Filtre Volume : ne trader que si volume > moyenne mobile du volume
    if use_volume_filter:
        df["volume_ma"] = df["Volume"].rolling(volume_ma_period).mean()
        vol_ok = df["Volume"] >= df["volume_ma"]
        df["signal_raw"] = np.where(vol_ok, df["signal_raw"], 0)

    # SHIFT ANTI-TRICHE : décaler d'une période
    df["signal"] = df["signal_raw"].shift(1)
    df["signal"] = df["signal"].fillna(0).astype(int)

    # DÉPART EN CASH : on ne suppose pas qu'on était en position avant les données
    # On force signal = 0 pendant le warmup des SMA (premières sma_slow périodes)
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
    """
    Mean Reversion : Bollinger Bands + RSI + Volume.
    - Achat : prix touche bande basse ET RSI < 30 (survente)
    - Vente : prix touche bande haute ou milieu (retour à la moyenne)
    - Filtre volume : confirme l'extrême
    """
    df = df.copy()
    upper, middle, lower = compute_bollinger_bands(df["Close"], bb_period, bb_std)
    df["BB_upper"] = upper
    df["BB_middle"] = middle
    df["BB_lower"] = lower
    df["RSI"] = compute_rsi(df["Close"], rsi_period)

    # Signal : 1 = long, -1 = short, 0 = cash
    touch_lower = df["Low"] <= df["BB_lower"]
    touch_upper = df["High"] >= df["BB_upper"]
    touch_middle = df["Close"] >= df["BB_middle"]

    # Entrée long : touche bande basse + RSI survente (shift 1 = on trade au bar suivant)
    entry_long = touch_lower & (df["RSI"] < rsi_oversold)
    exit_long = touch_upper | touch_middle

    if use_volume_filter:
        vol_ma = df["Volume"].rolling(volume_ma_period).mean()
        entry_long = entry_long & (df["Volume"] >= vol_ma)

    entry_long = entry_long.shift(1).fillna(False)

    # État machine
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
    """Buy & Hold : toujours en position long. Référence de base."""
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
    """
    MACD Crossover : achat quand MACD croise au-dessus de la ligne de signal.
    Stratégie de tendance, souvent performante en marchés directionnels.
    """
    df = df.copy()
    macd_line, signal_line, histogram = compute_macd(df["Close"], fast, slow, signal_period)
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = histogram

    # Croisement : MACD > signal → 1, sinon 0
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
    """
    RSI seul (naïf) : achète RSI < 30, vend RSI > 70.
    Souvent perdant : trop de whipsaws, pas de filtre de tendance.
    """
    df = df.copy()
    df["RSI"] = compute_rsi(df["Close"], rsi_period)
    df["signal_raw"] = np.where(df["RSI"] < rsi_oversold, 1, np.where(df["RSI"] > rsi_overbought, 0, np.nan))
    df["signal_raw"] = df["signal_raw"].ffill().fillna(0).astype(int)  # garde position jusqu'au prochain signal
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
    """
    Inverse SMA : achète quand SMA rapide < SMA lente (contrarian).
    Fait l'inverse du trend-following. Souvent catastrophique en tendance.
    """
    df = df.copy()
    df["SMA_fast"] = df["Close"].rolling(sma_fast).mean()
    df["SMA_slow"] = df["Close"].rolling(sma_slow).mean()
    df["signal_raw"] = np.where(df["SMA_fast"] < df["SMA_slow"], 1, 0)
    df["signal"] = df["signal_raw"].shift(1).fillna(0).astype(int)

    if start_in_cash:
        df.loc[df.index[:sma_slow], "signal"] = 0
    return df


# =============================================================================
# STRATÉGIE PORTFOLIO (style hedge fund)
# =============================================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR = moyenne des True Range sur period."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    return tr.rolling(period).mean()


def _compute_sharpe_weights(
    dfs: dict,
    symbols: list,
    common_idx: pd.Index,
    lookback: int,
    periods_per_year: int,
) -> list:
    """
    Allocation par Sharpe Ratio : weight_i = max(0, SR_i) / sum.
    Utilise un lookback rolling pour éviter le lookahead.
    """
    sharpes = []
    for sym in symbols:
        df = dfs[sym].reindex(common_idx).ffill().bfill()
        ret = df["strategy_return_net"].fillna(0) if "strategy_return_net" in df.columns else (df["signal"] * df["log_return"]).fillna(0)
        if len(ret) < lookback:
            sharpes.append(0.0)
            continue
        roll_mean = ret.rolling(lookback).mean()
        roll_std = ret.rolling(lookback).std().replace(0, np.nan)
        sr = np.sqrt(periods_per_year) * roll_mean / roll_std
        sr_final = sr.iloc[-1] if pd.notna(sr.iloc[-1]) else 0.0
        sharpes.append(max(0.0, float(sr_final)))
    total = sum(sharpes)
    if total <= 0:
        return [1.0 / len(symbols)] * len(symbols)
    return [s / total for s in sharpes]


def run_backtest_portfolio(
    symbols: list,
    weights: Optional[list] = None,
    allocation_method: str = "sharpe",
    sharpe_lookback: int = 504,
    rebalance_bars: int = 168,
    timeframe: str = "1h",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    sma_fast: int = 20,
    sma_slow: int = 50,
    use_rsi_filter: bool = True,
    rsi_period: int = 14,
    rsi_long_max: float = 70.0,
    rsi_short_min: float = 30.0,
    use_multi_timeframe: bool = True,
    mtf_timeframe: str = "4h",
    stop_loss_pct: Optional[float] = 5.0,
    take_profit_pct: Optional[float] = 15.0,
    max_dd_circuit_breaker: Optional[float] = 15.0,
    atr_filter_pct: Optional[float] = 95.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0002,
    volatility_filter: bool = True,
    vol_percentile: float = 90.0,
) -> Tuple[pd.DataFrame, dict]:
    """
    Portfolio multi-paires style hedge fund (version pro).
    - Allocation par Sharpe Ratio (ou manuelle)
    - Stop Loss / Take Profit par position
    - Multi-timeframe : confirmation tendance 4h
    - Circuit breaker : réduction exposition si DD > seuil
    - Filtre ATR : évite entrées en haute volatilité
    """
    periods_per_year = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365, "1wk": 52, "1mo": 12}.get(timeframe, 8760)

    dfs = {}
    dfs_mtf = {}
    for sym in symbols:
        ohlcv = fetch_ohlcv(sym, timeframe, since=since, until=until)
        df = clean_ohlcv(ohlcv)
        df = compute_log_returns(df)
        df = generate_signals(df, sma_fast, sma_slow, False, use_rsi_filter, rsi_period, rsi_long_max, rsi_short_min, False, 20, True)
        if use_multi_timeframe:
            ohlcv_mtf = fetch_ohlcv(sym, mtf_timeframe, since=since, until=until)
            df_mtf = clean_ohlcv(ohlcv_mtf)
            df_mtf["SMA_fast"] = df_mtf["Close"].rolling(sma_fast).mean()
            df_mtf["SMA_slow"] = df_mtf["Close"].rolling(sma_slow).mean()
            df_mtf["mtf_trend"] = (df_mtf["SMA_fast"] > df_mtf["SMA_slow"]).astype(int)
            df_mtf = df_mtf[["mtf_trend"]]
            df = df.join(df_mtf.reindex(df.index, method="ffill"), how="left")
            df["mtf_trend"] = df["mtf_trend"].fillna(0)
            df["signal"] = (df["signal"] * df["mtf_trend"]).astype(int)
        if atr_filter_pct is not None:
            df["ATR"] = _compute_atr(df, 14)
            atr_pct = df["ATR"] / df["Close"] * 100
            thresh = atr_pct.rolling(100).quantile(atr_filter_pct / 100).fillna(np.inf)
            df.loc[atr_pct > thresh, "signal"] = 0
        df = compute_strategy_returns(df, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
        df = apply_costs(df, commission_pct, slippage_pct)
        dfs[sym] = df

    common_idx = dfs[symbols[0]].index
    for sym in symbols[1:]:
        common_idx = common_idx.intersection(dfs[sym].index)
    common_idx = common_idx.sort_values()

    n = len(common_idx)
    portfolio_return = pd.Series(0.0, index=common_idx)
    bh_return = pd.Series(0.0, index=common_idx)
    eq_w = 1.0 / len(symbols)
    for sym in symbols:
        df = dfs[sym].reindex(common_idx).ffill().bfill()
        bh_return = bh_return + eq_w * df["log_return"].fillna(0)

    weight_series = {sym: pd.Series(eq_w, index=common_idx) for sym in symbols}
    if allocation_method == "sharpe" and n > sharpe_lookback:
        for k in range(sharpe_lookback, n, rebalance_bars):
            idx_slice = common_idx[:k]
            w = _compute_sharpe_weights(dfs, symbols, idx_slice, sharpe_lookback, periods_per_year)
            end_k = min(k + rebalance_bars, n)
            for j, sym in enumerate(symbols):
                weight_series[sym].iloc[k:end_k] = w[j]
    elif weights is not None and len(weights) == len(symbols) and abs(sum(weights) - 1.0) < 0.01:
        for sym, w in zip(symbols, weights):
            weight_series[sym] = pd.Series(w, index=common_idx)

    for sym in symbols:
        df = dfs[sym].reindex(common_idx).ffill().bfill()
        portfolio_return = portfolio_return + weight_series[sym] * df["strategy_return_net"].fillna(0)

    if volatility_filter and n > 50:
        roll_vol = portfolio_return.rolling(50).std().fillna(0)
        vol_thresh = roll_vol.quantile(vol_percentile / 100)
        vol_adj = np.where(roll_vol > vol_thresh, 0.5, 1.0)
        vol_adj = pd.Series(vol_adj, index=common_idx).fillna(1.0)
        portfolio_return = portfolio_return * vol_adj

    if max_dd_circuit_breaker is not None and n > 10:
        equity = np.exp(portfolio_return.cumsum())
        cummax = np.maximum.accumulate(equity)
        dd = (equity - cummax) / cummax * 100
        circuit_adj = np.where(dd < -max_dd_circuit_breaker, 0.5, 1.0)
        portfolio_return = pd.Series(portfolio_return.values * circuit_adj, index=common_idx)

    df_port = pd.DataFrame({"strategy_return": portfolio_return, "log_return": bh_return}, index=common_idx)
    df_port["strategy_equity"] = np.exp(portfolio_return.cumsum())
    df_port["bh_equity"] = np.exp(bh_return.cumsum())
    df_port["strategy_return_net"] = df_port["strategy_return"]
    df_port["strategy_equity_net"] = df_port["strategy_equity"]
    df_port["position_change"] = 0
    close_cols = []
    for i, sym in enumerate(symbols):
        col = f"Close_{sym.replace('/', '_')}"
        df_port[col] = dfs[sym].reindex(common_idx)["Close"].ffill().bfill()
        close_cols.append(col)
    norm_prices = [df_port[close_cols[i]] / df_port[close_cols[i]].iloc[0] * 100 for i in range(len(symbols))]
    w0 = [weight_series[sym].iloc[0] for sym in symbols]
    df_port["Close"] = sum(w0[i] * norm_prices[i] for i in range(len(symbols)))

    report = compute_risk_report(df_port, periods_per_year=periods_per_year)
    return df_port, report


# =============================================================================
# PORTFOLIO HF SENIOR (facteurs, Markowitz, VaR, corrélation)
# =============================================================================

def _factor_signal_momentum(returns: pd.Series, lookback: int = 20) -> pd.Series:
    """Momentum : rendement sur lookback, z-scoré."""
    mom = returns.rolling(lookback).sum()
    return (mom - mom.rolling(50).mean()) / mom.rolling(50).std().replace(0, np.nan)


def _factor_signal_value(close: pd.Series, lookback: int = 50) -> pd.Series:
    """Trend (Prix vs SMA) : z-score positif quand prix > SMA = tendance haussière = bullish.
    En crypto, la mean-reversion (acheter sous SMA) est souvent piégeuse ; on privilégie le trend."""
    sma = close.rolling(lookback).mean()
    std = close.rolling(lookback).std().replace(0, np.nan)
    return (close - sma) / std  # positif = bullish trend


def _factor_signal_volatility(returns: pd.Series, lookback: int = 20) -> pd.Series:
    """Low vol factor : inverse de la vol (low vol = score élevé)."""
    vol = returns.rolling(lookback).std().replace(0, np.nan)
    inv_vol = 1 / vol
    return (inv_vol - inv_vol.rolling(50).mean()) / inv_vol.rolling(50).std().replace(0, np.nan)


def _markowitz_max_sharpe(
    mu: np.ndarray,
    Sigma: np.ndarray,
    periods_per_year: int = 365,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Optimisation Markowitz : maximise le Sharpe ratio. Contraintes : w >= 0, sum(w) = 1, w_i <= max_weight."""
    n = len(mu)
    if n == 0 or Sigma.size == 0:
        return np.array([1.0 / max(1, n)] * max(1, n))

    def neg_sharpe(w):
        w = np.array(w)
        ret = w @ mu
        vol = np.sqrt(max(1e-12, w @ Sigma @ w))
        return -np.sqrt(periods_per_year) * ret / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, min(1.0, max_weight))] * n
    x0 = np.ones(n) / n
    try:
        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        return np.maximum(0, res.x) if res.success else np.ones(n) / n
    except Exception:
        return np.ones(n) / n


def _filter_correlated(
    symbols: list,
    ret_matrix: pd.DataFrame,
    max_correlation: float,
    factor_scores: dict,
) -> list:
    """Retire les actifs trop corrélés (garde celui avec le meilleur score facteur)."""
    if len(symbols) < 2 or max_correlation >= 1.0:
        return symbols
    corr = ret_matrix[symbols].corr()
    to_drop = set()
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i + 1 :]:
            if s1 in to_drop or s2 in to_drop:
                continue
            if abs(corr.loc[s1, s2]) > max_correlation:
                if factor_scores.get(s1, 0) >= factor_scores.get(s2, 0):
                    to_drop.add(s2)
                else:
                    to_drop.add(s1)
    return [s for s in symbols if s not in to_drop]


def _detect_market_regime(bh_return: pd.Series, lookback: int = 63, bull_thresh: float = 0.02, bear_thresh: float = -0.02) -> pd.Series:
    """
    Régime de marché : 1=bull, -1=bear, 0=range.
    Basé sur le rendement roulant annualisé du B&H.
    """
    roll_ret = bh_return.rolling(lookback).sum()
    ann_factor = np.sqrt(252 / lookback) if lookback > 0 else 1
    roll_ann = roll_ret * ann_factor
    regime = pd.Series(0, index=bh_return.index)
    regime[roll_ann > bull_thresh] = 1
    regime[roll_ann < bear_thresh] = -1
    return regime.fillna(0)


def _apply_vol_targeting(returns: pd.Series, target_vol_pct: float, window: int = 20, periods_per_year: int = 365) -> pd.Series:
    """Vol targeting : scale les rendements pour viser une vol annualisée cible."""
    if target_vol_pct <= 0 or window < 5:
        return returns
    target_vol = target_vol_pct / 100 / np.sqrt(periods_per_year)
    roll_vol = returns.rolling(window).std().replace(0, np.nan).ffill().bfill()
    scale = target_vol / (roll_vol + 1e-10)
    scale = scale.clip(0.5, 2.0)
    return returns * scale


def _markowitz_max_ir(
    mu: np.ndarray,
    Sigma: np.ndarray,
    bench_ret: float,
    periods_per_year: int = 365,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Maximise l'Information Ratio (surperformance vs benchmark)."""
    n = len(mu)
    if n == 0 or Sigma.size == 0:
        return np.ones(n) / n

    def neg_ir(w):
        w = np.array(w)
        ret = float(w @ mu)
        active = ret - bench_ret
        te = np.sqrt(max(1e-12, w @ Sigma @ w))
        if te < 1e-10:
            return 0.0
        return -np.sqrt(periods_per_year) * active / te

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, min(1.0, max_weight))] * n
    x0 = np.ones(n) / n
    try:
        res = minimize(neg_ir, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        return np.maximum(0, res.x) if res.success else np.ones(n) / n
    except Exception:
        return np.ones(n) / n


def run_backtest_portfolio_hf(
    symbols: list,
    timeframe: str = "1d",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    factor_momentum_lb: int = 20,
    factor_value_lb: int = 50,
    factor_vol_lb: int = 20,
    factor_w_momentum: float = 0.5,
    factor_w_value: float = 0.35,
    factor_w_vol: float = 0.15,
    factor_threshold: float = -0.1,
    markowitz_lookback: int = 252,
    rebalance_bars: int = 21,
    var_limit_pct: Optional[float] = 2.0,
    max_correlation: float = 0.85,
    max_weight_per_asset: float = 0.4,
    max_dd_circuit_breaker: Optional[float] = 12.0,
    use_regime_detection: bool = True,
    regime_lookback: int = 63,
    regime_bear_scale: float = 0.5,
    regime_range_scale: float = 0.75,
    vol_target_ann_pct: Optional[float] = 15.0,
    vol_target_window: int = 20,
    turnover_threshold: Optional[float] = 0.05,
    max_turnover_per_rebalance: Optional[float] = 0.3,
    use_ir_objective: bool = False,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0002,
    spread_pct: float = 0.0,
    data_source: str = "yahoo",
) -> Tuple[pd.DataFrame, dict]:
    """
    Portfolio HF Senior : stratégie complexe type hedge fund.
    - Facteurs quantitatifs : Momentum, Trend (prix vs SMA), Low Vol
    - Combinaison pondérée → score par actif
    - Filtre : garde actifs avec score > threshold
    - Markowitz : max Sharpe sur actifs filtrés
    - Contraintes : max poids, max corrélation, VaR, circuit breaker DD
    - Objectif : battre le Buy & Hold en optimisant le Sharpe
    """
    periods_per_year = {"1d": 365, "1wk": 52, "1mo": 12}.get(timeframe, 365)
    warmup = max(markowitz_lookback, factor_value_lb) + 100
    since_fetch = since
    until_fetch = until
    if since is not None and until is not None:
        tf_days = {"1d": 1, "1wk": 7, "1mo": 30}.get(timeframe, 1)
        since_fetch = since - timedelta(days=warmup * tf_days)

    dfs = {}
    for sym in symbols:
        ohlcv = fetch_ohlcv(sym, timeframe, since=since_fetch, until=until_fetch, data_source=data_source)
        df = clean_ohlcv(ohlcv)
        df = compute_log_returns(df)
        mom = _factor_signal_momentum(df["log_return"], factor_momentum_lb)
        val = _factor_signal_value(df["Close"], factor_value_lb)
        vol = _factor_signal_volatility(df["log_return"], factor_vol_lb)
        df["factor_score"] = (
            factor_w_momentum * mom.fillna(0)
            + factor_w_value * val.fillna(0)
            + factor_w_vol * vol.fillna(0)
        )
        dfs[sym] = df

    common_idx = dfs[symbols[0]].index
    for sym in symbols[1:]:
        common_idx = common_idx.intersection(dfs[sym].index)
    common_idx = common_idx.sort_values()
    n = len(common_idx)

    eq_w = 1.0 / len(symbols)
    bh_return = pd.Series(0.0, index=common_idx)
    for sym in symbols:
        bh_return = bh_return + eq_w * dfs[sym].reindex(common_idx)["log_return"].fillna(0)

    # S&P 500 comme benchmark marché (^GSPC)
    sp500_return = pd.Series(0.0, index=common_idx)
    try:
        ohlcv_sp = fetch_ohlcv("^GSPC", timeframe, since=since_fetch, until=until_fetch, data_source=data_source)
        if ohlcv_sp:
            df_sp = clean_ohlcv(ohlcv_sp)
            df_sp = compute_log_returns(df_sp)
            sp500_return = df_sp["log_return"].reindex(common_idx).fillna(0)
    except Exception:
        pass

    ret_matrix = pd.DataFrame({sym: dfs[sym].reindex(common_idx)["log_return"].fillna(0) for sym in symbols})

    regime_scale = pd.Series(1.0, index=common_idx)
    if use_regime_detection and regime_lookback > 0:
        regime = _detect_market_regime(bh_return, regime_lookback)
        regime_scale[regime == -1] = regime_bear_scale
        regime_scale[regime == 0] = regime_range_scale

    weight_series = {sym: pd.Series(0.0, index=common_idx) for sym in symbols}
    prev_weights = {sym: 0.0 for sym in symbols}

    for k in range(markowitz_lookback, n, rebalance_bars):
        end_k = min(k + rebalance_bars, n)
        window = ret_matrix.iloc[k - markowitz_lookback : k]
        bh_window = bh_return.iloc[k - markowitz_lookback : k]
        bench_ret = float(bh_window.mean())

        factor_scores_k = {}
        for sym in symbols:
            sc = dfs[sym].reindex(common_idx)["factor_score"].fillna(-10).iloc[k]
            factor_scores_k[sym] = float(sc)

        selected = [s for s in symbols if factor_scores_k.get(s, -10) > factor_threshold]
        if len(selected) == 0:
            for sym in symbols:
                weight_series[sym].iloc[k:end_k] = 0
                prev_weights[sym] = 0.0
            continue

        selected = _filter_correlated(selected, window, max_correlation, factor_scores_k)
        if len(selected) == 0:
            continue

        mu = window[selected].mean().values
        Sigma = window[selected].cov().values
        Sigma = np.nan_to_num(Sigma, nan=0, posinf=0, neginf=0)
        if Sigma.size > 0:
            Sigma = Sigma + np.eye(len(selected)) * 1e-6

        if use_ir_objective:
            w = _markowitz_max_ir(mu, Sigma, bench_ret, periods_per_year, max_weight_per_asset)
        else:
            w = _markowitz_max_sharpe(mu, Sigma, periods_per_year, max_weight_per_asset)
        w = np.maximum(0, w)
        w = w / (w.sum() + 1e-12)

        if var_limit_pct is not None and var_limit_pct > 0:
            port_ret = (window[selected] * w).sum(axis=1)
            var95 = float(np.percentile(port_ret, 5) * 100)
            if abs(var95) > var_limit_pct:
                scale = var_limit_pct / (abs(var95) + 1e-8)
                w = w * min(1.0, scale)

        w_dict = {selected[i]: float(w[i]) for i in range(len(selected))}
        for sym in symbols:
            if sym not in w_dict:
                w_dict[sym] = 0.0

        if turnover_threshold is not None and turnover_threshold > 0:
            turnover = sum(abs(w_dict[s] - prev_weights.get(s, 0)) for s in symbols)
            if turnover < turnover_threshold:
                w_dict = prev_weights.copy()

        if max_turnover_per_rebalance is not None and max_turnover_per_rebalance > 0:
            turnover = sum(abs(w_dict[s] - prev_weights.get(s, 0)) for s in symbols)
            if turnover > max_turnover_per_rebalance:
                blend = max_turnover_per_rebalance / (turnover + 1e-10)
                for s in symbols:
                    w_dict[s] = prev_weights.get(s, 0) + blend * (w_dict[s] - prev_weights.get(s, 0))

        prev_weights = w_dict.copy()
        reg_scale = float(regime_scale.iloc[k])
        for sym in symbols:
            weight_series[sym].iloc[k:end_k] = w_dict[sym] * reg_scale

    portfolio_return = pd.Series(0.0, index=common_idx)
    for sym in symbols:
        portfolio_return = portfolio_return + weight_series[sym] * ret_matrix[sym]

    if vol_target_ann_pct is not None and vol_target_ann_pct > 0:
        portfolio_return = _apply_vol_targeting(portfolio_return, vol_target_ann_pct, vol_target_window, periods_per_year)

    portfolio_return_brute = portfolio_return.copy()
    if commission_pct > 0 or slippage_pct > 0 or spread_pct > 0:
        turnover = pd.Series(0.0, index=common_idx)
        for sym in symbols:
            turnover = turnover + weight_series[sym].diff().abs().fillna(0)
        # Même convention que apply_costs : commission + slippage + demi-spread par trade
        cost_per_trade = commission_pct + slippage_pct + spread_pct / 2
        portfolio_return = portfolio_return - turnover * cost_per_trade

    if max_dd_circuit_breaker is not None and max_dd_circuit_breaker > 0 and n > 10:
        equity = np.exp(portfolio_return.cumsum().values)
        cummax = np.maximum.accumulate(equity)
        dd_pct = (equity - cummax) / (cummax + 1e-12) * 100
        circuit = np.where(dd_pct < -max_dd_circuit_breaker, 0.5, 1.0)
        portfolio_return = pd.Series(portfolio_return.values * circuit, index=common_idx)
        portfolio_return_brute = pd.Series(portfolio_return_brute.values * circuit, index=common_idx)

    df_port = pd.DataFrame({"strategy_return": portfolio_return, "log_return": bh_return, "sp500_log_return": sp500_return}, index=common_idx)
    df_port["strategy_equity"] = np.exp(portfolio_return_brute.cumsum())
    df_port["bh_equity"] = np.exp(bh_return.cumsum())
    df_port["sp500_equity"] = np.exp(sp500_return.cumsum())
    df_port["strategy_return_net"] = df_port["strategy_return"]
    df_port["strategy_equity_net"] = np.exp(portfolio_return.cumsum())
    close_cols = []
    pos_changes = pd.Series(0.0, index=common_idx)
    for sym in symbols:
        col = f"Close_{sym.replace('/', '_')}"
        df_port[col] = dfs[sym].reindex(common_idx)["Close"].ffill().bfill()
        sig_col = f"signal_{sym.replace('/', '_')}"
        df_port[sig_col] = (weight_series[sym] > 0.01).astype(int)
        pos_changes = pos_changes + df_port[sig_col].diff().abs().fillna(0)
        close_cols.append(col)
    df_port["position_change"] = pos_changes
    turnover_series = pd.Series(0.0, index=common_idx)
    for sym in symbols:
        turnover_series = turnover_series + weight_series[sym].diff().abs().fillna(0)
    df_port["turnover"] = turnover_series
    norm_prices = [df_port[c] / df_port[c].iloc[0] * 100 for c in close_cols]
    w0 = [weight_series[sym].iloc[min(markowitz_lookback, n - 1)] for sym in symbols]
    df_port["Close"] = sum(w0[i] * norm_prices[i] for i in range(len(symbols)))

    if since is not None and until is not None and len(df_port) > 0:
        since_ts = pd.Timestamp(since)
        until_ts = pd.Timestamp(until)
        mask = (df_port.index >= since_ts) & (df_port.index <= until_ts)
        df_port = df_port.loc[mask]
        if len(df_port) > 0:
            eq0 = df_port["strategy_equity_net"].iloc[0]
            eq0_brute = df_port["strategy_equity"].iloc[0]
            bh0 = df_port["bh_equity"].iloc[0]
            sp0 = df_port["sp500_equity"].iloc[0]
            df_port["strategy_equity_net"] = df_port["strategy_equity_net"] / eq0
            df_port["strategy_equity"] = df_port["strategy_equity"] / eq0_brute
            df_port["bh_equity"] = df_port["bh_equity"] / bh0
            df_port["sp500_equity"] = df_port["sp500_equity"] / sp0

    report = compute_risk_report(df_port, periods_per_year=periods_per_year)
    return df_port, report


def get_live_signals(
    symbols: list,
    timeframe: str = "1d",
    sma_fast: int = 20,
    sma_slow: int = 50,
    use_rsi_filter: bool = True,
    rsi_period: int = 14,
    rsi_long_max: float = 70.0,
    rsi_short_min: float = 30.0,
    limit: int = 100,
) -> dict:
    """
    Récupère les données via Yahoo Finance et calcule les signaux actuels.
    Retourne : {symbol: {price, signal, rsi, sma_fast, sma_slow, last_update}}
    """
    from datetime import datetime, timedelta
    until = datetime.now()
    since = until - timedelta(days=limit * 2)
    results = {}
    for sym in symbols:
        try:
            ohlcv = fetch_ohlcv_yahoo(sym, timeframe, limit=limit, since=since, until=until)
            df = clean_ohlcv(ohlcv)
            df = compute_log_returns(df)
            df = generate_signals(df, sma_fast, sma_slow, False, use_rsi_filter, rsi_period, rsi_long_max, rsi_short_min, False, 20, True)
            last = df.iloc[-1]
            rsi_val = float(last["RSI"]) if "RSI" in df.columns and pd.notna(last.get("RSI")) else None
            results[sym] = {
                "price": float(last["Close"]),
                "signal": int(last["signal"]),
                "rsi": rsi_val,
                "sma_fast": float(last["SMA_fast"]) if "SMA_fast" in df.columns and pd.notna(last.get("SMA_fast")) else None,
                "sma_slow": float(last["SMA_slow"]) if "SMA_slow" in df.columns and pd.notna(last.get("SMA_slow")) else None,
                "last_update": df.index[-1],
            }
        except Exception as e:
            results[sym] = {"error": str(e)}
    return results


# =============================================================================
# ÉTAPE 4 : PnL ENGINE (Calcul des gains/pertes)
# =============================================================================

def compute_strategy_returns(
    df: pd.DataFrame,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Rendement stratégie = signal * rendement actif
    Si stop_loss_pct / take_profit_pct fournis : applique SL/TP (sortie anticipée).
    """
    df = df.copy()

    if stop_loss_pct is not None or take_profit_pct is not None:
        # Simulation bar-by-bar avec SL/TP
        sl = stop_loss_pct or 1.0
        tp = take_profit_pct or 10.0
        strategy_returns = []
        entry_price = None
        position = 0

        for i in range(len(df)):
            sig = int(df["signal"].iloc[i])
            row = df.iloc[i]
            ret = 0.0

            # Entrée : nouvelle position
            if sig != 0 and position == 0:
                position = sig
                entry_price = row["Open"]

            if position != 0 and entry_price is not None:
                low, high = row["Low"], row["High"]
                if position == 1:  # long
                    if stop_loss_pct and low <= entry_price * (1 - sl / 100):
                        ret = np.log((1 - sl / 100))
                        position = 0
                    elif take_profit_pct and high >= entry_price * (1 + tp / 100):
                        ret = np.log(1 + tp / 100)
                        position = 0
                    else:
                        ret = row["log_return"]
                        if sig == 0:
                            position = 0
                elif position == -1:  # short
                    if stop_loss_pct and high >= entry_price * (1 + sl / 100):
                        ret = np.log(1 - sl / 100)
                        position = 0
                    elif take_profit_pct and low <= entry_price * (1 - tp / 100):
                        ret = np.log(1 + tp / 100)
                        position = 0
                    else:
                        ret = -row["log_return"]
                        if sig == 0:
                            position = 0

            strategy_returns.append(ret)

        df["strategy_return"] = strategy_returns
    else:
        df["strategy_return"] = df["signal"] * df["log_return"]
        df["strategy_return"] = df["strategy_return"].fillna(0)

    # Cumul des log-returns → performance brute
    df["cum_log_return_strategy"] = df["strategy_return"].cumsum()
    df["strategy_equity"] = np.exp(df["cum_log_return_strategy"])

    # Buy & Hold pour comparaison
    df["cum_log_return_bh"] = df["log_return"].cumsum()
    df["bh_equity"] = np.exp(df["cum_log_return_bh"])

    return df


# =============================================================================
# ÉTAPE 5 : REALITY CHECK (Frais et Slippage)
# =============================================================================

def apply_costs(
    df: pd.DataFrame,
    commission_pct: float = 0.001,   # 0.10% Binance / 0.05-0.10% actions
    slippage_pct: float = 0.0002,   # 0.02% crypto / 0.01-0.03% actions
    spread_pct: float = 0.0,        # Demi-spread par trade, même unité que commission (ex: 0.0002 = 0.02%)
) -> pd.DataFrame:
    """
    Détecte chaque changement de position (trade).
    Pénalité = commission + slippage + spread/2 par trade.
    Buy & Hold : 1 achat au début + 1 vente à la fin (2 trades).
    """
    if df.empty or len(df) == 0:
        return df
    df = df.copy()
    df["position_change"] = df["signal"].diff().abs()

    cost_per_trade = commission_pct + slippage_pct + spread_pct / 2

    # Buy & Hold : signal constant = 1 → 1 entrée (début) + 1 sortie (fin)
    if (df["signal"] == 1).all():
        df["trade_cost"] = 0.0
        df.loc[df.index[0], "trade_cost"] = cost_per_trade   # Achat initial
        df.loc[df.index[-1], "trade_cost"] = cost_per_trade  # Vente finale
        df["position_change"] = 0.0
        df.loc[df.index[0], "position_change"] = 1.0
        df.loc[df.index[-1], "position_change"] = 1.0
    else:
        trades = df["position_change"] > 0
        df["trade_cost"] = np.where(trades, cost_per_trade, 0)
        # Fin de période : si encore en position, on vend (1 trade de sortie)
        if df["signal"].iloc[-1] != 0:
            df.loc[df.index[-1], "trade_cost"] = df.loc[df.index[-1], "trade_cost"] + cost_per_trade
            df.loc[df.index[-1], "position_change"] = df.loc[df.index[-1], "position_change"] + 1.0
    df["strategy_return_net"] = df["strategy_return"] - df["trade_cost"]

    # Recalcul de l'equity nette
    df["cum_log_return_strategy_net"] = df["strategy_return_net"].cumsum()
    df["strategy_equity_net"] = np.exp(df["cum_log_return_strategy_net"])

    return df


# =============================================================================
# ÉTAPE 6 : RISK REPORT (Statistiques de Trader)
# =============================================================================

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """
    Sharpe = E[R_p - R_f] / σ_p
    """
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Maximum Drawdown : plus grosse chute du capital.
    Retourne (drawdown_pct, date_peak, date_trough)
    """
    if equity_curve.empty or len(equity_curve) == 0:
        return 0.0, None, None
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / (cummax + 1e-12)
    if drawdown.empty:
        return 0.0, None, None
    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    peak_idx = equity_curve[:trough_idx].idxmax()
    return float(max_dd), peak_idx, trough_idx


def win_rate(df: pd.DataFrame) -> float:
    """
    Pourcentage de trades gagnants.
    Un trade = période entre deux changements de position (entry → exit).
    """
    trade_indices = df.index[df["position_change"] > 0].tolist()
    if len(trade_indices) < 2:
        return 0.0
    equity = df["strategy_equity_net"]
    winners = 0
    for i in range(len(trade_indices) - 1):
        start_idx = trade_indices[i]
        end_idx = trade_indices[i + 1]
        ret = (equity.loc[end_idx] / equity.loc[start_idx]) - 1
        if ret > 0:
            winners += 1
    return winners / (len(trade_indices) - 1) * 100


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    VaR historique : perte maximale attendue avec (1-confidence)% de probabilité.
    Ex: VaR 95% = 5% du temps la perte sera pire que ce seuil.
    """
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    ES (Expected Shortfall / CVaR) : perte moyenne dans les pires (1-confidence)% des cas.
    Plus conservateur que la VaR : prend en compte la "queue" des pertes.
    """
    var = var_historical(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """
    Sortino = E[R - Rf] / σ_downside
    Ne pénalise que la volatilité à la baisse (plus pertinent que Sharpe).
    """
    excess = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / downside.std()


def calmar_ratio(
    returns: pd.Series,
    equity: pd.Series,
    periods_per_year: int = 8760,
) -> float:
    """
    Calmar = Rendement annualisé / Max Drawdown
    Mesure le rendement par unité de risque (drawdown).
    """
    total_return = equity.iloc[-1] - 1
    n_periods = len(returns)
    if n_periods < 2:
        return 0.0
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    md, _, _ = max_drawdown(equity)
    if md == 0:
        return 0.0
    return annual_return / abs(md)


def profit_factor(df: pd.DataFrame) -> float:
    """
    Profit Factor = Somme des gains / |Somme des pertes|
    > 1 = stratégie profitable, < 1 = perdante.
    """
    ret = df["strategy_return_net"]
    gains = ret[ret > 0].sum()
    losses = ret[ret < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return gains / abs(losses)


# =============================================================================
# ANALYSES PRO : Monte Carlo, Bootstrap, Tests statistiques
# =============================================================================

def monte_carlo_simulation(
    returns: pd.Series,
    n_simulations: int = 1000,
    periods_per_year: int = 8760,
    seed: Optional[int] = None,
) -> dict:
    """
    Monte Carlo : rééchantillonnage AVEC remise (bootstrap) des rendements.
    Chaque simulation = un tirage aléatoire de n rendements (certains répétés, d'autres absents).
    Donne une vraie distribution : proba entre 0% et 100%.
    """
    if seed is not None:
        np.random.seed(seed)
    ret = returns.values
    n = len(ret)
    if n < 10:
        return {"error": "Pas assez de données"}

    final_equities = []
    max_dds = []
    sharpes = []

    for _ in range(n_simulations):
        # Rééchantillonnage AVEC remise (pas permutation) → somme varie à chaque sim
        sampled = np.random.choice(ret, size=n, replace=True)
        equity = np.exp(np.cumsum(sampled))
        final_equities.append(equity[-1] - 1)
        cummax = np.maximum.accumulate(equity)
        dd = (equity - cummax) / cummax
        max_dds.append(dd.min() * 100)
        if sampled.std() > 0:
            sharpes.append(np.sqrt(periods_per_year) * sampled.mean() / sampled.std())
        else:
            sharpes.append(0.0)

    final_equities = np.array(final_equities) * 100
    max_dds = np.array(max_dds)
    sharpes = np.array(sharpes)

    return {
        "final_return_mean_pct": float(np.mean(final_equities)),
        "final_return_std_pct": float(np.std(final_equities)),
        "final_return_median_pct": float(np.median(final_equities)),
        "final_return_5pct": float(np.percentile(final_equities, 5)),
        "final_return_95pct": float(np.percentile(final_equities, 95)),
        "max_dd_mean_pct": float(np.mean(max_dds)),
        "max_dd_5pct": float(np.percentile(max_dds, 5)),
        "max_dd_95pct": float(np.percentile(max_dds, 95)),
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_5pct": float(np.percentile(sharpes, 5)),
        "sharpe_95pct": float(np.percentile(sharpes, 95)),
        "prob_positive_return": float(np.mean(final_equities > 0) * 100),
        "n_simulations": n_simulations,
    }


def bootstrap_analysis(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    periods_per_year: int = 8760,
    seed: Optional[int] = None,
) -> dict:
    """
    Bootstrap : rééchantillonnage par blocs (préserve l'autocorrélation).
    Donne intervalles de confiance pour le rendement et le Sharpe.
    """
    if seed is not None:
        np.random.seed(seed)
    ret = returns.values
    n = len(ret)
    if n < block_size * 2:
        return {"error": "Pas assez de données"}

    n_blocks = n // block_size
    total_returns = []
    total_sharpes = []

    for _ in range(n_bootstrap):
        blocks = [ret[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
        sampled_blocks = np.random.choice(n_blocks, n_blocks, replace=True)
        sampled_ret = np.concatenate([blocks[b] for b in sampled_blocks])
        total_returns.append((np.exp(np.sum(sampled_ret)) - 1) * 100)
        if sampled_ret.std() > 0:
            total_sharpes.append(np.sqrt(periods_per_year) * sampled_ret.mean() / sampled_ret.std())
        else:
            total_sharpes.append(0.0)

    return {
        "return_mean_pct": float(np.mean(total_returns)),
        "return_ci_95_low": float(np.percentile(total_returns, 2.5)),
        "return_ci_95_high": float(np.percentile(total_returns, 97.5)),
        "sharpe_mean": float(np.mean(total_sharpes)),
        "sharpe_ci_95_low": float(np.percentile(total_sharpes, 2.5)),
        "sharpe_ci_95_high": float(np.percentile(total_sharpes, 97.5)),
        "n_bootstrap": n_bootstrap,
    }


def statistical_tests(returns: pd.Series) -> dict:
    """
    Tests statistiques sur la distribution des rendements :
    - Normalité (Jarque-Bera, Shapiro-Wilk)
    - Skewness (asymétrie)
    - Kurtosis (queues épaisses)
    """
    from scipy import stats

    ret = returns.dropna().values
    if len(ret) < 8:
        return {"error": "Pas assez de données pour les tests"}

    jb_stat, jb_pvalue = stats.jarque_bera(ret)
    sw_stat, sw_pvalue = stats.shapiro(ret) if len(ret) <= 5000 else (np.nan, np.nan)
    skew = float(stats.skew(ret))
    kurt = float(stats.kurtosis(ret))

    return {
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "normal_distribution": jb_pvalue > 0.05,
        "shapiro_stat": float(sw_stat) if not np.isnan(sw_stat) else None,
        "shapiro_pvalue": float(sw_pvalue) if not np.isnan(sw_pvalue) else None,
        "skewness": skew,
        "kurtosis": kurt,
        "interpretation": (
            "Returns close to normal distribution" if jb_pvalue > 0.05
            else "Returns NOT normal (fat tails, asymmetry)"
        ),
    }


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 8760,
) -> float:
    """
    Information Ratio = (R_strat - R_bench) / Tracking Error
    Mesure la surperformance par unité de risque actif.
    """
    active = strategy_returns - benchmark_returns
    if active.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * active.mean() / active.std()


def rolling_metrics(
    df: pd.DataFrame,
    window: int = 252,
    periods_per_year: int = 8760,
) -> pd.DataFrame:
    """
    Métriques glissantes : Sharpe, Drawdown, Volatilité sur fenêtre roulante.
    """
    ret = df["strategy_return_net"]
    eq = df["strategy_equity_net"]
    roll = pd.DataFrame(index=df.index)
    roll["rolling_sharpe"] = ret.rolling(window).apply(
        lambda x: np.sqrt(periods_per_year) * x.mean() / x.std() if x.std() > 0 else 0
    )
    roll["rolling_vol"] = ret.rolling(window).std() * np.sqrt(periods_per_year) * 100
    cummax = eq.rolling(window).max()
    roll["rolling_dd"] = (eq - cummax) / cummax * 100
    return roll.dropna(how="all")


def stress_test(
    df: pd.DataFrame,
    n_worst_periods: int = 5,
    window: int = 24,
) -> dict:
    """
    Stress test : identifie les pires périodes (fenêtre glissante).
    Simule : "Et si j'avais commencé au pire moment ?"
    """
    eq = df["strategy_equity_net"]
    rolling_return = eq.pct_change(window).dropna()
    worst = rolling_return.nsmallest(n_worst_periods)

    worst_list = []
    for idx in worst.index:
        pos = df.index.get_loc(idx)
        start_idx = max(0, pos - window)
        worst_list.append({
            "start": str(df.index[start_idx]),
            "end": str(idx),
            "return_pct": float(rolling_return.loc[idx] * 100),
        })

    return {
        "worst_periods": worst_list,
        "worst_single_return_pct": float(rolling_return.min() * 100),
    }


def compute_risk_report(
    df: pd.DataFrame,
    periods_per_year: int = 8760,
) -> dict:
    """
    Rapport de risque complet : Sharpe, Sortino, Calmar, VaR, ES, Profit Factor, etc.
    """
    if df.empty or len(df) == 0:
        return {
            "sharpe_ratio": 0.0, "sharpe_ratio_bh": 0.0, "avg_turnover_pct": 0.0,
            "sortino_ratio": 0.0, "calmar_ratio": 0.0, "profit_factor": 0.0,
            "information_ratio": 0.0, "max_drawdown_pct": 0.0, "max_dd_peak_date": None,
            "max_dd_trough_date": None, "var_95_pct": 0.0, "var_99_pct": 0.0,
            "es_95_pct": 0.0, "es_99_pct": 0.0, "win_rate_pct": 0.0, "period_win_rate_pct": 0.0,
            "n_trades": 0, "total_return_pct": 0.0, "bh_return_pct": 0.0,
            "volatility_ann_pct": 0.0, "bh_volatility_ann_pct": 0.0,
            "bh_max_drawdown_pct": 0.0, "bh_win_rate_pct": 0.0,
        }
    returns = df["strategy_return_net"].dropna()
    equity = df["strategy_equity_net"]

    sharpe = sharpe_ratio(returns, periods_per_year=periods_per_year)
    bh_returns = df["log_return"].dropna().reindex(returns.index).fillna(0)
    sharpe_bh = sharpe_ratio(bh_returns, periods_per_year=periods_per_year) if len(bh_returns) > 1 and bh_returns.std() > 1e-10 else 0.0
    bh_vol = float(bh_returns.std() * np.sqrt(periods_per_year) * 100) if len(bh_returns) > 1 else 0.0
    bh_equity = df["bh_equity"] if "bh_equity" in df.columns else np.exp(bh_returns.cumsum())
    md_bh, _, _ = max_drawdown(bh_equity)
    bh_wr = float((bh_returns > 0).mean() * 100) if len(bh_returns) > 0 else 0.0
    sortino = sortino_ratio(returns, periods_per_year=periods_per_year)
    md, peak, trough = max_drawdown(equity)
    calmar = calmar_ratio(returns, equity, periods_per_year)
    wr = win_rate(df)
    period_wr = float((returns > 0).mean() * 100) if len(returns) > 0 else 0.0
    pf = profit_factor(df)

    # VaR et ES (en % de rendement par période)
    var_95 = var_historical(returns, 0.95) * 100
    var_99 = var_historical(returns, 0.99) * 100
    es_95 = expected_shortfall(returns, 0.95) * 100
    es_99 = expected_shortfall(returns, 0.99) * 100

    n_trades = (df["position_change"] > 0).sum()
    avg_turnover_pct = float(df["turnover"].mean() * 100) if "turnover" in df.columns else 0.0

    info_ratio = information_ratio(
        returns, df["log_return"].dropna().reindex(returns.index).fillna(0),
        periods_per_year=periods_per_year,
    )

    out = {
        "sharpe_ratio": sharpe,
        "sharpe_ratio_bh": sharpe_bh,
        "avg_turnover_pct": avg_turnover_pct,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "profit_factor": pf,
        "information_ratio": info_ratio,
        "max_drawdown_pct": md * 100,
        "max_dd_peak_date": peak,
        "max_dd_trough_date": trough,
        "var_95_pct": var_95,
        "var_99_pct": var_99,
        "es_95_pct": es_95,
        "es_99_pct": es_99,
        "win_rate_pct": wr,
        "period_win_rate_pct": period_wr,
        "n_trades": n_trades,
        "total_return_pct": (equity.iloc[-1] - 1) * 100,
        "bh_return_pct": (df["bh_equity"].iloc[-1] - 1) * 100,
        "volatility_ann_pct": returns.std() * np.sqrt(periods_per_year) * 100,
        "bh_volatility_ann_pct": bh_vol,
        "bh_max_drawdown_pct": md_bh * 100,
        "bh_win_rate_pct": bh_wr,
    }
    if "sp500_equity" in df.columns and "sp500_log_return" in df.columns:
        sp500_ret = df["sp500_log_return"].dropna().reindex(returns.index).fillna(0)
        sp500_eq = df["sp500_equity"]
        out["sp500_return_pct"] = (sp500_eq.iloc[-1] - 1) * 100
        out["sp500_sharpe"] = sharpe_ratio(sp500_ret, periods_per_year=periods_per_year) if len(sp500_ret) > 1 and sp500_ret.std() > 1e-10 else 0.0
        md_sp, _, _ = max_drawdown(sp500_eq)
        out["sp500_max_drawdown_pct"] = md_sp * 100
        out["sp500_volatility_ann_pct"] = float(sp500_ret.std() * np.sqrt(periods_per_year) * 100) if len(sp500_ret) > 1 else 0.0
        out["sp500_win_rate_pct"] = float((sp500_ret > 0).mean() * 100) if len(sp500_ret) > 0 else 0.0
        out["information_ratio_vs_sp500"] = information_ratio(returns, sp500_ret, periods_per_year=periods_per_year)
    return out


def run_pro_analysis(
    df: pd.DataFrame,
    periods_per_year: int = 8760,
    n_monte_carlo: int = 500,
    n_bootstrap: int = 500,
) -> dict:
    """
    Lance toutes les analyses pro : Monte Carlo, Bootstrap, tests statistiques,
    stress test, rolling metrics.
    """
    returns = df["strategy_return_net"].dropna()
    if len(returns) < 50:
        return {"error": "Pas assez de données (min 50 périodes)"}

    monte = monte_carlo_simulation(returns, n_simulations=n_monte_carlo, periods_per_year=periods_per_year)
    bootstrap = bootstrap_analysis(returns, n_bootstrap=n_bootstrap, periods_per_year=periods_per_year)
    stats = statistical_tests(returns)
    stress = stress_test(df, n_worst_periods=5, window=min(24, len(df) // 10))
    roll = rolling_metrics(df, window=min(252, len(df) // 4), periods_per_year=periods_per_year)

    return {
        "monte_carlo": monte,
        "bootstrap": bootstrap,
        "statistical_tests": stats,
        "stress_test": stress,
        "rolling_metrics": roll,
    }


# =============================================================================
# ÉTAPE 7 : VISUALIZER (Graphiques professionnels)
# =============================================================================

def get_price_chart_figure(
    df: pd.DataFrame,
    title: str = "Prix + Signaux",
    show_signals: bool = True,
    strategy_type: str = "sma",
    dark_theme: bool = False,
):
    """
    Graphique : Prix + indicateurs (SMA ou Bollinger) + marqueurs d'entrée/sortie.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    price_color = "#e2e8f0" if dark_theme else "black"
    ax.plot(df.index, df["Close"], label="Price", color=price_color, alpha=0.85, linewidth=1)

    if strategy_type == "price":
        pass  # Prix seul, pas d'indicateur
    elif strategy_type == "bollinger" and "BB_upper" in df.columns:
        ax.plot(df.index, df["BB_upper"], label="Bande haute", color="red", alpha=0.6, linewidth=1)
        ax.plot(df.index, df["BB_middle"], label="Bande milieu", color="gray", alpha=0.6, linewidth=1)
        ax.plot(df.index, df["BB_lower"], label="Bande basse", color="green", alpha=0.6, linewidth=1)
    elif strategy_type == "macd" and "MACD" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df.index, df["MACD"], label="MACD", color="blue", alpha=0.8, linewidth=1.5)
        ax2.plot(df.index, df["MACD_signal"], label="Signal", color="orange", alpha=0.8, linewidth=1)
        ax2.set_ylabel("MACD")
        ax2.legend(loc="upper right", fontsize=8)
    elif strategy_type == "rsi" and "RSI" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df.index, df["RSI"], label="RSI", color="purple", alpha=0.8, linewidth=1)
        ax2.axhline(y=30, color="green", linestyle="--", alpha=0.5)
        ax2.axhline(y=70, color="red", linestyle="--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right", fontsize=8)
    elif "SMA_fast" in df.columns:
        ax.plot(df.index, df["SMA_fast"], label="Fast SMA", color="green", alpha=0.8, linewidth=1.5)
        ax.plot(df.index, df["SMA_slow"], label="Slow SMA", color="orange", alpha=0.8, linewidth=1.5)

    if show_signals and "signal" in df.columns:
        # Points d'entrée (passage à 1)
        entries = df[df["signal"].diff() == 1]
        exits = df[df["signal"].diff() == -1]
        if len(entries) > 0:
            ax.scatter(entries.index, entries["Close"], color="green", marker="^", s=80, zorder=5, label="Long entry")
        if len(exits) > 0:
            ax.scatter(exits.index, exits["Close"], color="red", marker="v", s=80, zorder=5, label="Exit")

    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    if dark_theme:
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#e2e8f0")
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                t.set_color("#e2e8f0")
    plt.tight_layout()
    return fig


def get_plot_figure(df: pd.DataFrame, title: str = "Backtest: Performance & Drawdown", symbol: str = "", close_cols: Optional[list] = None, dark_theme: bool = False, compact: bool = False, force_show_brute: bool = False, hide_comparison: bool = False):
    """
    Returns a matplotlib figure. compact=True: Performance + Drawdown only (no price).
    """
    if compact or (close_cols and len(close_cols) > 3):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True)
        axes = [axes[0], axes[1]]
        skip_price = True
    else:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[1, 2, 1], sharex=True)
        skip_price = False
    if dark_theme:
        fig.patch.set_facecolor("#0f172a")
        line_colors = ["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#22d3ee", "#fb923c", "#4ade80"]
    else:
        line_colors = list(plt.cm.tab10.colors)

    # --- Chart 1 : Courbe du prix (contexte) ---
    if not skip_price:
        ax0 = axes[0]
        if close_cols and len(close_cols) > 0:
            colors = line_colors if dark_theme else list(plt.cm.tab10.colors)
            for i, col in enumerate(close_cols):
                lbl = col.replace("Close_", "").replace("_", "/")
                ax0.plot(df.index, df[col], label=lbl, color=colors[i % len(colors)], alpha=0.8, linewidth=1.5)
                sig_col = f"signal_{col.replace('Close_', '')}"
                if sig_col in df.columns:
                    sig = df[sig_col].fillna(0).astype(int)
                    entries = df[sig.diff() == 1]
                    exits = df[sig.diff() == -1]
                    max_markers = 120
                    if len(entries) > max_markers:
                        entries = entries.iloc[:: max(1, len(entries) // max_markers)]
                    if len(exits) > max_markers:
                        exits = exits.iloc[:: max(1, len(exits) // max_markers)]
                    if len(entries) > 0:
                        ax0.scatter(entries.index, entries[col], color="green", marker="^", s=50, zorder=5, alpha=0.85)
                    if len(exits) > 0:
                        ax0.scatter(exits.index, exits[col], color="red", marker="v", s=50, zorder=5, alpha=0.85)
        else:
            c0 = "#e2e8f0" if dark_theme else "black"
            ax0.plot(df.index, df["Close"], label=f"Price {symbol or 'asset'}", color=c0, alpha=0.85, linewidth=1.5)
            if "signal" in df.columns:
                entries = df[df["signal"].diff() == 1]
                exits = df[df["signal"].diff() == -1]
                if len(entries) > 0:
                    ax0.scatter(entries.index, entries["Close"], color="green", marker="^", s=80, zorder=5, label="Buy")
                if len(exits) > 0:
                    ax0.scatter(exits.index, exits["Close"], color="red", marker="v", s=80, zorder=5, label="Sell")
        ax0.set_ylabel("Price ($)")
        ax0.set_title("Price evolution")
        ax0.legend(loc="upper left", fontsize=9)
        ax0.grid(True, alpha=0.3)
        ax0.set_ylim(bottom=0)
        ax1 = axes[1]
    else:
        ax1 = axes[0]

    # --- Performance ---
    strat_c = "#60a5fa" if dark_theme else "darkblue"
    if not hide_comparison:
        bh_c = "#94a3b8" if dark_theme else "gray"
        sp500_c = "#64748b" if dark_theme else "#475569"
        ax1.plot(df.index, df["bh_equity"], label="B&H (portfolio)", color=bh_c, alpha=0.85, linewidth=1.5)
        if "sp500_equity" in df.columns:
            ax1.plot(df.index, df["sp500_equity"], label="S&P 500 (benchmark)", color=sp500_c, alpha=0.85, linewidth=1.5, linestyle="-.")
        show_brute = force_show_brute or not np.allclose(df["strategy_equity"].values, df["strategy_equity_net"].values, rtol=1e-5)
        if show_brute:
            ax1.plot(df.index, df["strategy_equity"], label="Gross strat. (no fees)", color="darkorange", alpha=0.95, linewidth=2, linestyle="--", zorder=3)
    ax1.plot(df.index, df["strategy_equity_net"], label="Strategy" if hide_comparison else "Strategy (with fees)", color=strat_c, linewidth=2)
    # Marqueurs achat/vente
    if hide_comparison and len(df) >= 2:
        ax1.scatter(df.index[0], df["strategy_equity_net"].iloc[0], color="green", marker="^", s=60, zorder=5, alpha=0.9, label="Buy")
        ax1.scatter(df.index[-1], df["strategy_equity_net"].iloc[-1], color="red", marker="v", s=60, zorder=5, alpha=0.9, label="Sell")
    elif "signal" in df.columns and not hide_comparison:
        sig = df["signal"].fillna(0).astype(int)
        entries = df[sig.diff() == 1]
        exits = df[sig.diff() == -1]
        max_m = 150
        for evt, label, color, m in [(entries, "Buy", "green", "^"), (exits, "Sell", "red", "v")]:
            if len(evt) > 0:
                sub = evt.iloc[:: max(1, len(evt) // max_m)] if len(evt) > max_m else evt
                ax1.scatter(sub.index, sub["strategy_equity_net"], color=color, marker=m, s=45, zorder=5, alpha=0.9, label=label)
    elif "turnover" in df.columns and not hide_comparison:
        rebal = df[df["turnover"] > 0.001]
        if len(rebal) > 0:
            sub = rebal.iloc[:: max(1, len(rebal) // 120)] if len(rebal) > 120 else rebal
            ax1.scatter(sub.index, sub["strategy_equity_net"], color="purple", marker="o", s=35, zorder=5, alpha=0.8, label="Rebalance")
    ax1.set_ylabel("Performance (× capital)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # --- Drawdown ---
    ax2 = axes[2] if not skip_price else axes[1]
    cummax = df["strategy_equity_net"].cummax()
    drawdown = (df["strategy_equity_net"] - cummax) / cummax
    ax2.fill_between(df.index, drawdown, 0, color="#ef4444" if dark_theme else "crimson", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(top=0)
    if dark_theme:
        all_axes = [ax1, ax2] if skip_price else axes
        for ax in all_axes:
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#475569")
            ax.xaxis.label.set_color("#e2e8f0")
            ax.yaxis.label.set_color("#e2e8f0")
            ax.title.set_color("#e2e8f0")
            if ax.get_legend():
                for t in ax.get_legend().get_texts():
                    t.set_color("#e2e8f0")
    plt.tight_layout()
    return fig


def plot_results(df: pd.DataFrame, output_path: str = "backtest_results.png"):
    """Saves the chart to disk."""
    fig = get_plot_figure(df)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {output_path}")


# =============================================================================
# WALK-FORWARD (In-Sample / Out-of-Sample)
# =============================================================================

def run_backtest_on_df(
    df_raw: pd.DataFrame,
    sma_fast: int,
    sma_slow: int,
    short_allowed: bool,
    start_in_cash: bool,
    use_rsi_filter: bool,
    rsi_period: int,
    rsi_long_max: float,
    rsi_short_min: float,
    use_volume_filter: bool,
    volume_ma_period: int,
    commission_pct: float,
    slippage_pct: float,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    periods_per_year: int = 8760,
    spread_pct: float = 0.0,
) -> Tuple[pd.DataFrame, dict]:
    """
    Lance le backtest sur un DataFrame déjà nettoyé (pour walk-forward).
    """
    df = df_raw.copy()
    df = compute_log_returns(df)
    df = generate_signals(
        df, sma_fast, sma_slow, short_allowed,
        use_rsi_filter, rsi_period, rsi_long_max, rsi_short_min,
        use_volume_filter, volume_ma_period, start_in_cash,
    )
    df = compute_strategy_returns(df, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
    df = apply_costs(df, commission_pct, slippage_pct, spread_pct)
    report = compute_risk_report(df, periods_per_year=periods_per_year)
    return df, report


def run_backtest_on_df_bollinger(
    df_raw: pd.DataFrame,
    bb_period: int,
    bb_std: float,
    rsi_period: int,
    rsi_oversold: float,
    rsi_overbought: float,
    short_allowed: bool,
    start_in_cash: bool,
    use_volume_filter: bool,
    volume_ma_period: int,
    commission_pct: float,
    slippage_pct: float,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    periods_per_year: int = 8760,
) -> Tuple[pd.DataFrame, dict]:
    """
    Lance le backtest Bollinger Bands sur un DataFrame déjà nettoyé.
    """
    df = df_raw.copy()
    df = compute_log_returns(df)
    df = generate_signals_bollinger(
        df,
        bb_period=bb_period,
        bb_std=bb_std,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        use_volume_filter=use_volume_filter,
        volume_ma_period=volume_ma_period,
        short_allowed=short_allowed,
        start_in_cash=start_in_cash,
    )
    df = compute_strategy_returns(df, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
    df = apply_costs(df, commission_pct, slippage_pct)
    report = compute_risk_report(df, periods_per_year=periods_per_year)
    return df, report


def run_backtest_on_df_buy_hold(
    df_raw: pd.DataFrame,
    commission_pct: float,
    slippage_pct: float,
    periods_per_year: int = 8760,
    spread_pct: float = 0.0,
) -> Tuple[pd.DataFrame, dict]:
    if df_raw.empty or len(df_raw) == 0:
        return df_raw, compute_risk_report(df_raw, periods_per_year=periods_per_year)
    df = df_raw.copy()
    df = compute_log_returns(df)
    df = generate_signals_buy_hold(df)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct, spread_pct)
    return df, compute_risk_report(df, periods_per_year=periods_per_year)


def run_backtest_on_df_macd(
    df_raw: pd.DataFrame,
    macd_fast: int, macd_slow: int, macd_signal: int,
    commission_pct: float, slippage_pct: float,
    periods_per_year: int = 8760,
) -> Tuple[pd.DataFrame, dict]:
    df = df_raw.copy()
    df = compute_log_returns(df)
    df = generate_signals_macd(df, fast=macd_fast, slow=macd_slow, signal_period=macd_signal)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct)
    return df, compute_risk_report(df, periods_per_year=periods_per_year)


def run_backtest_on_df_rsi_naive(
    df_raw: pd.DataFrame,
    rsi_period: int, rsi_oversold: float, rsi_overbought: float,
    commission_pct: float, slippage_pct: float,
    periods_per_year: int = 8760,
) -> Tuple[pd.DataFrame, dict]:
    df = df_raw.copy()
    df = compute_log_returns(df)
    df = generate_signals_rsi_naive(df, rsi_period, rsi_oversold, rsi_overbought)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct)
    return df, compute_risk_report(df, periods_per_year=periods_per_year)


def run_backtest_on_df_inverse_sma(
    df_raw: pd.DataFrame,
    sma_fast: int, sma_slow: int,
    commission_pct: float, slippage_pct: float,
    periods_per_year: int = 8760,
) -> Tuple[pd.DataFrame, dict]:
    df = df_raw.copy()
    df = compute_log_returns(df)
    df = generate_signals_inverse_sma(df, sma_fast, sma_slow)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct)
    return df, compute_risk_report(df, periods_per_year=periods_per_year)


def run_walk_forward_backtest(
    symbol: str,
    timeframe: str,
    since: datetime,
    until: datetime,
    in_sample_pct: float = 0.6,
    sma_fast: int = 20,
    sma_slow: int = 50,
    short_allowed: bool = False,
    start_in_cash: bool = True,
    use_rsi_filter: bool = False,
    rsi_period: int = 14,
    rsi_long_max: float = 70.0,
    rsi_short_min: float = 30.0,
    use_volume_filter: bool = False,
    volume_ma_period: int = 20,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0002,
    spread_pct: float = 0.0,
    data_source: str = "yahoo",
) -> dict:
    """
    Walk-Forward : split In-Sample (optimisation) / Out-of-Sample (validation).
    Confirme que la stratégie n'était pas juste de la chance sur la période.
    """
    warmup_bars = max(sma_slow, rsi_period if use_rsi_filter else 0, volume_ma_period if use_volume_filter else 0)
    since_fetch, until_fetch = since, until
    if warmup_bars > 0:
        tf_days = {"1d": 1, "1wk": 7, "1mo": 30}.get(timeframe, 1)
        since_fetch = since - timedelta(days=warmup_bars * tf_days)
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since_fetch, until=until_fetch, data_source=data_source)
    df = clean_ohlcv(ohlcv_raw)

    since_ts = pd.Timestamp(since)
    until_ts = pd.Timestamp(until)
    valid_mask = (df.index >= since_ts) & (df.index <= until_ts)
    valid_indices = df.index[valid_mask].tolist()
    n_valid = len(valid_indices)
    if n_valid == 0:
        if len(df) == 0:
            raise ValueError(f"No data for ticker in period [{since.date()}, {until.date()}]. Check ticker and dates.")
        data_min, data_max = df.index.min(), df.index.max()
        raise ValueError(
            f"No data in period [{since.date()}, {until.date()}]. "
            f"Available data: {data_min.date()} to {data_max.date()}. "
            f"Start date may be after latest data, or end date may be in the future."
        )
    split_in_valid = min(int(n_valid * in_sample_pct), n_valid - 1)
    split_pos = df.index.get_loc(valid_indices[split_in_valid])

    df_in = df.iloc[: split_pos + 1]
    df_out_raw = df.iloc[max(0, split_pos + 1 - warmup_bars):] if warmup_bars > 0 else df.iloc[split_pos + 1:]

    periods_per_year = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365, "1wk": 52, "1mo": 12}.get(timeframe, 8760)

    df_in_bt, report_in = run_backtest_on_df(
        df_in,
        sma_fast=sma_fast, sma_slow=sma_slow, short_allowed=short_allowed,
        start_in_cash=start_in_cash,
        use_rsi_filter=use_rsi_filter, rsi_period=rsi_period,
        rsi_long_max=rsi_long_max, rsi_short_min=rsi_short_min,
        use_volume_filter=use_volume_filter, volume_ma_period=volume_ma_period,
        commission_pct=commission_pct, slippage_pct=slippage_pct,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        periods_per_year=periods_per_year, spread_pct=spread_pct,
    )
    df_out_bt, report_out = run_backtest_on_df(
        df_out_raw,
        sma_fast=sma_fast, sma_slow=sma_slow, short_allowed=short_allowed,
        start_in_cash=start_in_cash,
        use_rsi_filter=use_rsi_filter, rsi_period=rsi_period,
        rsi_long_max=rsi_long_max, rsi_short_min=rsi_short_min,
        use_volume_filter=use_volume_filter, volume_ma_period=volume_ma_period,
        commission_pct=commission_pct, slippage_pct=slippage_pct,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        periods_per_year=periods_per_year, spread_pct=spread_pct,
    )

    # Filter out-of-sample to validation period only (exclude warmup from report)
    oos_start_ts = valid_indices[split_in_valid] if split_in_valid < n_valid else df.index[split_pos]
    df_out_bt = df_out_bt[df_out_bt.index >= oos_start_ts] if len(df_out_bt) > 0 else df_out_bt
    report_out = compute_risk_report(df_out_bt, periods_per_year=periods_per_year) if len(df_out_bt) > 0 else report_out

    return {
        "in_sample": {
            "df": df_in_bt,
            "report": report_in,
            "period": (df_in.index[0], df_in.index[-1]),
        },
        "out_of_sample": {
            "df": df_out_bt,
            "report": report_out,
            "period": (oos_start_ts, df.index[-1]) if len(df_out_bt) > 0 else (None, None),
        },
        "robust": report_in["total_return_pct"] > 0 and report_out["total_return_pct"] > 0,
    }


def run_walk_forward_backtest_bollinger(
    symbol: str,
    timeframe: str,
    since: datetime,
    until: datetime,
    in_sample_pct: float = 0.6,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    short_allowed: bool = False,
    start_in_cash: bool = True,
    use_volume_filter: bool = True,
    volume_ma_period: int = 20,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0002,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
) -> dict:
    """
    Walk-Forward pour stratégie Bollinger Bands.
    """
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)

    split_idx = int(len(df) * in_sample_pct)
    df_in = df.iloc[:split_idx]
    df_out = df.iloc[split_idx:]

    periods_per_year = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365, "1wk": 52, "1mo": 12}.get(timeframe, 8760)

    df_in_bt, report_in = run_backtest_on_df_bollinger(
        df_in,
        bb_period=bb_period, bb_std=bb_std,
        rsi_period=rsi_period, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought,
        short_allowed=short_allowed, start_in_cash=start_in_cash,
        use_volume_filter=use_volume_filter, volume_ma_period=volume_ma_period,
        commission_pct=commission_pct, slippage_pct=slippage_pct,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        periods_per_year=periods_per_year,
    )
    df_out_bt, report_out = run_backtest_on_df_bollinger(
        df_out,
        bb_period=bb_period, bb_std=bb_std,
        rsi_period=rsi_period, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought,
        short_allowed=short_allowed, start_in_cash=start_in_cash,
        use_volume_filter=use_volume_filter, volume_ma_period=volume_ma_period,
        commission_pct=commission_pct, slippage_pct=slippage_pct,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        periods_per_year=periods_per_year,
    )

    return {
        "in_sample": {"df": df_in_bt, "report": report_in, "period": (df_in.index[0], df_in.index[-1])},
        "out_of_sample": {"df": df_out_bt, "report": report_out, "period": (df_out.index[0], df_out.index[-1])},
        "robust": report_in["total_return_pct"] > 0 and report_out["total_return_pct"] > 0,
    }


def run_walk_forward_backtest_buy_hold(symbol, timeframe, since, until, in_sample_pct=0.6, commission_pct=0.001, slippage_pct=0.0002, spread_pct=0.0, data_source="yahoo"):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since, until=until, data_source=data_source)
    df = clean_ohlcv(ohlcv_raw)
    if df.empty or len(df) < 2:
        raise ValueError(f"Insufficient data for Walk-Forward: {len(df)} bars. Check ticker ({symbol}) and period ({since.date()} → {until.date()}).")
    split_idx = max(1, min(len(df) - 1, int(len(df) * in_sample_pct)))
    ppy = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365, "1wk": 52, "1mo": 12}.get(timeframe, 8760)
    df_in_bt, r_in = run_backtest_on_df_buy_hold(df.iloc[:split_idx], commission_pct, slippage_pct, periods_per_year=ppy, spread_pct=spread_pct)
    df_out_bt, r_out = run_backtest_on_df_buy_hold(df.iloc[split_idx:], commission_pct, slippage_pct, periods_per_year=ppy, spread_pct=spread_pct)
    return {"in_sample": {"df": df_in_bt, "report": r_in, "period": (df.index[0], df.iloc[split_idx - 1].name)},
            "out_of_sample": {"df": df_out_bt, "report": r_out, "period": (df.iloc[split_idx].name, df.index[-1])},
            "robust": r_in["total_return_pct"] > 0 and r_out["total_return_pct"] > 0}


def run_walk_forward_backtest_macd(symbol, timeframe, since, until, in_sample_pct=0.6, macd_fast=12, macd_slow=26, macd_signal=9, commission_pct=0.001, slippage_pct=0.0002):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    split_idx = int(len(df) * in_sample_pct)
    ppy = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}.get(timeframe, 8760)
    df_in_bt, r_in = run_backtest_on_df_macd(df.iloc[:split_idx], macd_fast, macd_slow, macd_signal, commission_pct, slippage_pct, ppy)
    df_out_bt, r_out = run_backtest_on_df_macd(df.iloc[split_idx:], macd_fast, macd_slow, macd_signal, commission_pct, slippage_pct, ppy)
    return {"in_sample": {"df": df_in_bt, "report": r_in, "period": (df.index[0], df.iloc[split_idx - 1].name)},
            "out_of_sample": {"df": df_out_bt, "report": r_out, "period": (df.iloc[split_idx].name, df.index[-1])},
            "robust": r_in["total_return_pct"] > 0 and r_out["total_return_pct"] > 0}


def run_walk_forward_backtest_rsi_naive(symbol, timeframe, since, until, in_sample_pct=0.6, rsi_period=14, rsi_oversold=30, rsi_overbought=70, commission_pct=0.001, slippage_pct=0.0002):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    split_idx = int(len(df) * in_sample_pct)
    ppy = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}.get(timeframe, 8760)
    df_in_bt, r_in = run_backtest_on_df_rsi_naive(df.iloc[:split_idx], rsi_period, rsi_oversold, rsi_overbought, commission_pct, slippage_pct, ppy)
    df_out_bt, r_out = run_backtest_on_df_rsi_naive(df.iloc[split_idx:], rsi_period, rsi_oversold, rsi_overbought, commission_pct, slippage_pct, ppy)
    return {"in_sample": {"df": df_in_bt, "report": r_in, "period": (df.index[0], df.iloc[split_idx - 1].name)},
            "out_of_sample": {"df": df_out_bt, "report": r_out, "period": (df.iloc[split_idx].name, df.index[-1])},
            "robust": r_in["total_return_pct"] > 0 and r_out["total_return_pct"] > 0}


def run_walk_forward_backtest_inverse_sma(symbol, timeframe, since, until, in_sample_pct=0.6, sma_fast=20, sma_slow=50, commission_pct=0.001, slippage_pct=0.0002):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    split_idx = int(len(df) * in_sample_pct)
    ppy = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}.get(timeframe, 8760)
    df_in_bt, r_in = run_backtest_on_df_inverse_sma(df.iloc[:split_idx], sma_fast, sma_slow, commission_pct, slippage_pct, ppy)
    df_out_bt, r_out = run_backtest_on_df_inverse_sma(df.iloc[split_idx:], sma_fast, sma_slow, commission_pct, slippage_pct, ppy)
    return {"in_sample": {"df": df_in_bt, "report": r_in, "period": (df.index[0], df.iloc[split_idx - 1].name)},
            "out_of_sample": {"df": df_out_bt, "report": r_out, "period": (df.iloc[split_idx].name, df.index[-1])},
            "robust": r_in["total_return_pct"] > 0 and r_out["total_return_pct"] > 0}


def _optimize_portfolio_hf_factors(
    symbols: list,
    timeframe: str,
    since: datetime,
    until: datetime,
    base_kwargs: dict,
    optimize_objective: str = "ir",
    turnover_penalty: float = 0.1,
) -> dict:
    """
    Optimise facteurs, seuil, lookbacks, rebalance sur In-Sample.
    Objectif : IR (Information Ratio) ou Sharpe, avec pénalité turnover.
    """
    mom_lb_opts = [20, 50]
    val_lb_opts = [50, 100]
    mk_lb_opts = [126, 252]
    rebal_opts = [10, 21]

    best_score = -np.inf
    best_kw = base_kwargs.copy()

    for mom_lb in mom_lb_opts:
        for val_lb in val_lb_opts:
            for mk_lb in mk_lb_opts:
                if mk_lb < val_lb + 30:
                    continue
                for rebal in rebal_opts:
                    kw = {
                        **base_kwargs,
                        "factor_momentum_lb": mom_lb,
                        "factor_value_lb": val_lb,
                        "factor_vol_lb": min(30, mom_lb),
                        "markowitz_lookback": mk_lb,
                        "rebalance_bars": rebal,
                        "use_ir_objective": optimize_objective == "ir",
                    }
                    try:
                        _, report = run_backtest_portfolio_hf(symbols=symbols, timeframe=timeframe, since=since, until=until, **kw)
                        ir = report.get("information_ratio", 0) or 0
                        sharpe = report.get("sharpe_ratio", 0) or 0
                        turnover = report.get("avg_turnover_pct", 0) or 0
                        if optimize_objective == "ir":
                            score = ir - turnover_penalty * turnover / 100
                        else:
                            score = sharpe - turnover_penalty * turnover / 100
                        if score > best_score:
                            best_score = score
                            best_kw = kw.copy()
                    except Exception:
                        pass

    x0 = [
        best_kw.get("factor_w_momentum", 0.5),
        best_kw.get("factor_w_value", 0.35),
        best_kw.get("factor_threshold", -0.1),
    ]
    bounds = [(0.1, 0.8), (0.1, 0.8), (-0.5, 0.5)]
    constraints = [{"type": "ineq", "fun": lambda x: 0.9 - x[0] - x[1]}]

    def neg_obj(x):
        w_mom, w_val, thresh = float(x[0]), float(x[1]), float(x[2])
        w_vol = max(0.05, 1.0 - w_mom - w_val)
        total = w_mom + w_val + w_vol
        kw = {**best_kw, "factor_w_momentum": w_mom / total, "factor_w_value": w_val / total, "factor_w_vol": w_vol / total, "factor_threshold": thresh}
        try:
            _, report = run_backtest_portfolio_hf(symbols=symbols, timeframe=timeframe, since=since, until=until, **kw)
            ir = report.get("information_ratio", 0) or 0
            sharpe = report.get("sharpe_ratio", 0) or 0
            turnover = report.get("avg_turnover_pct", 0) or 0
            if optimize_objective == "ir":
                score = ir - turnover_penalty * turnover / 100
            else:
                score = sharpe - turnover_penalty * turnover / 100
            return -score
        except Exception:
            return 1e6

    try:
        res = minimize(neg_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            w_mom, w_val, thresh = float(res.x[0]), float(res.x[1]), float(res.x[2])
            w_vol = max(0.05, 1.0 - w_mom - w_val)
            total = w_mom + w_val + w_vol
            best_kw.update({
                "factor_w_momentum": w_mom / total,
                "factor_w_value": w_val / total,
                "factor_w_vol": w_vol / total,
                "factor_threshold": thresh,
            })
    except Exception:
        pass
    return best_kw


def run_walk_forward_backtest_portfolio_hf(
    symbols: list,
    timeframe: str,
    since: datetime,
    until: datetime,
    in_sample_pct: float = 0.6,
    optimize_factors: bool = True,
    rolling_windows: int = 1,
    optimize_objective: str = "ir",
    turnover_penalty: float = 0.1,
    **kwargs,
) -> dict:
    """
    Walk-Forward Portfolio HF.
    - rolling_windows=1 : split unique In/Out.
    - rolling_windows>1 : fenêtre glissante, plusieurs validations.
    """
    ohlcv = fetch_ohlcv(symbols[0], timeframe, since=since, until=until)
    df_ref = clean_ohlcv(ohlcv)
    n_total = len(df_ref)
    split_idx = int(n_total * in_sample_pct)

    if rolling_windows <= 1:
        split_date = df_ref.index[split_idx]
        if optimize_factors:
            opt_kw = _optimize_portfolio_hf_factors(
                symbols, timeframe, since, split_date, kwargs,
                optimize_objective=optimize_objective,
                turnover_penalty=turnover_penalty,
            )
        else:
            opt_kw = kwargs

        df_in, r_in = run_backtest_portfolio_hf(symbols=symbols, timeframe=timeframe, since=since, until=split_date, **opt_kw)
        df_out, r_out = run_backtest_portfolio_hf(symbols=symbols, timeframe=timeframe, since=split_date, until=until, **opt_kw)

        return {
            "in_sample": {"df": df_in, "report": r_in, "period": (df_ref.index[0], df_ref.iloc[split_idx - 1])},
            "out_of_sample": {"df": df_out, "report": r_out, "period": (df_ref.iloc[split_idx], df_ref.index[-1])},
            "robust": r_in["total_return_pct"] > 0 and r_out["total_return_pct"] > 0,
            "optimized_params": opt_kw if optimize_factors else None,
        }

    n_out = n_total - split_idx
    step = max(1, n_out // rolling_windows) if rolling_windows > 1 else n_out
    oos_reports = []
    opt_kw = kwargs

    for i in range(rolling_windows):
        start_oos = split_idx + i * step
        end_oos = min(start_oos + step, n_total)
        if start_oos >= end_oos:
            break
        start_is = max(0, start_oos - split_idx)
        end_is = start_oos
        since_is = df_ref.index[start_is]
        until_is = df_ref.index[end_is - 1]
        since_oos = df_ref.index[start_oos]
        until_oos = df_ref.index[end_oos - 1]

        if optimize_factors:
            opt_kw = _optimize_portfolio_hf_factors(
                symbols, timeframe, since_is, until_is, kwargs,
                optimize_objective=optimize_objective,
                turnover_penalty=turnover_penalty,
            )

        df_out, r_out = run_backtest_portfolio_hf(symbols=symbols, timeframe=timeframe, since=since_oos, until=until_oos, **opt_kw)
        oos_reports.append(r_out)

    r_out_avg = oos_reports[0].copy() if oos_reports else {}
    for k in ["total_return_pct", "sharpe_ratio", "information_ratio", "bh_return_pct", "max_drawdown_pct", "sortino_ratio", "win_rate_pct"]:
        if oos_reports and k in oos_reports[0]:
            r_out_avg[k] = np.mean([r.get(k, 0) for r in oos_reports])
    df_in, r_in = run_backtest_portfolio_hf(symbols=symbols, timeframe=timeframe, since=since, until=df_ref.index[split_idx - 1], **opt_kw)
    df_out_full, _ = run_backtest_portfolio_hf(symbols=symbols, timeframe=timeframe, since=df_ref.index[split_idx], until=until, **opt_kw)

    return {
        "in_sample": {"df": df_in, "report": r_in, "period": (df_ref.index[0], df_ref.iloc[split_idx - 1])},
        "out_of_sample": {"df": df_out_full, "report": r_out_avg, "period": (df_ref.iloc[split_idx], df_ref.index[-1])},
        "robust": all(r["total_return_pct"] > 0 for r in oos_reports),
        "optimized_params": opt_kw if optimize_factors else None,
        "rolling_reports": oos_reports,
    }


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_backtest(
    symbol: str = "AAPL",
    timeframe: str = "1h",
    limit: int = 1000,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    sma_fast: int = 20,
    sma_slow: int = 50,
    short_allowed: bool = False,
    start_in_cash: bool = True,
    use_rsi_filter: bool = False,
    rsi_period: int = 14,
    rsi_long_max: float = 70.0,
    rsi_short_min: float = 30.0,
    use_volume_filter: bool = False,
    volume_ma_period: int = 20,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0002,
    spread_pct: float = 0.0,
    data_source: str = "yahoo",
    plot: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Exécute le backtest complet et retourne (DataFrame, rapport de risque).
    since/until : période de test (optionnel). Si fourni, récupère les données sur cette plage.
    start_in_cash : True = on part de zéro, pas de position héritée.
    verbose : False = pas de print (évite les conflits avec Streamlit).
    """
    def _log(msg):
        if verbose:
            try:
                print(msg)
            except (ValueError, OSError):
                pass

    # 1. Data Pipeline (with warmup before start date for indicator initialization)
    _log("Étape 1 : Récupération des données...")
    warmup_bars = max(sma_slow, rsi_period if use_rsi_filter else 0, volume_ma_period if use_volume_filter else 0) if (since is not None and until is not None) else 0
    since_fetch, until_fetch = since, until
    if since is not None and until is not None and warmup_bars > 0:
        tf_days = {"1d": 1, "1wk": 7, "1mo": 30}.get(timeframe, 1)
        since_fetch = since - timedelta(days=warmup_bars * tf_days)
    ohlcv_raw = fetch_ohlcv(
        symbol, timeframe,
        limit=limit,
        since=since_fetch,
        until=until_fetch,
        data_source=data_source,
    )

    # 2. Data Cleaning
    _log("Étape 2 : Nettoyage des données...")
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)

    # 3. Strategy
    _log("Étape 3 : Génération des signaux...")
    df = generate_signals(
        df,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        short_allowed=short_allowed,
        start_in_cash=start_in_cash,
        use_rsi_filter=use_rsi_filter,
        rsi_period=rsi_period,
        rsi_long_max=rsi_long_max,
        rsi_short_min=rsi_short_min,
        use_volume_filter=use_volume_filter,
        volume_ma_period=volume_ma_period,
    )

    # 4. PnL
    _log("Étape 4 : Calcul des performances...")
    df = compute_strategy_returns(df, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)

    # 5. Reality Check
    _log("Étape 5 : Application des frais et slippage...")
    df = apply_costs(df, commission_pct, slippage_pct, spread_pct)

    # 5b. Filter to user period (exclude warmup from metrics)
    if since is not None and until is not None and warmup_bars > 0 and len(df) > 0:
        since_ts, until_ts = pd.Timestamp(since), pd.Timestamp(until)
        mask = (df.index >= since_ts) & (df.index <= until_ts)
        df = df.loc[mask]

    # 6. Risk Report
    _log("Étape 6 : Rapport de risque...")
    periods_per_year = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365, "1wk": 52, "1mo": 12}.get(timeframe, 8760)
    report = compute_risk_report(df, periods_per_year=periods_per_year)

    # 7. Visualizer
    if plot:
        _log("Étape 7 : Génération du graphique...")
        plot_results(df)

    return df, report


def run_backtest_bollinger(
    symbol: str = "AAPL",
    timeframe: str = "1h",
    limit: int = 1000,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    short_allowed: bool = False,
    start_in_cash: bool = True,
    use_volume_filter: bool = True,
    volume_ma_period: int = 20,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0002,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    plot: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Exécute le backtest Bollinger Bands et retourne (DataFrame, rapport de risque).
    """
    def _log(msg):
        if verbose:
            try:
                print(msg)
            except (ValueError, OSError):
                pass

    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)
    df = generate_signals_bollinger(
        df,
        bb_period=bb_period,
        bb_std=bb_std,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        use_volume_filter=use_volume_filter,
        volume_ma_period=volume_ma_period,
        short_allowed=short_allowed,
        start_in_cash=start_in_cash,
    )
    df = compute_strategy_returns(df, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
    df = apply_costs(df, commission_pct, slippage_pct)
    periods_per_year = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365, "1wk": 52, "1mo": 12}.get(timeframe, 8760)
    report = compute_risk_report(df, periods_per_year=periods_per_year)

    if plot:
        fig = get_price_chart_figure(df, title=f"{symbol} | Bollinger BB{bb_period}", strategy_type="bollinger")
        plt.show()

    return df, report


def run_backtest_buy_hold(symbol="AAPL", timeframe="1d", limit=1000, since=None, until=None, commission_pct=0.001, slippage_pct=0.0002, spread_pct=0.0, data_source="yahoo", plot=True, verbose=True):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until, data_source=data_source)
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)
    df = generate_signals_buy_hold(df)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct, spread_pct)
    ppy = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365, "1wk": 52, "1mo": 12}.get(timeframe, 8760)
    return df, compute_risk_report(df, periods_per_year=ppy)


def run_backtest_macd(symbol="AAPL", timeframe="1d", limit=1000, since=None, until=None, macd_fast=12, macd_slow=26, macd_signal=9, commission_pct=0.001, slippage_pct=0.0002, plot=True, verbose=True):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)
    df = generate_signals_macd(df, fast=macd_fast, slow=macd_slow, signal_period=macd_signal)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct)
    ppy = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}.get(timeframe, 8760)
    return df, compute_risk_report(df, periods_per_year=ppy)


def run_backtest_rsi_naive(symbol="AAPL", timeframe="1d", limit=1000, since=None, until=None, rsi_period=14, rsi_oversold=30, rsi_overbought=70, commission_pct=0.001, slippage_pct=0.0002, plot=True, verbose=True):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)
    df = generate_signals_rsi_naive(df, rsi_period, rsi_oversold, rsi_overbought)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct)
    ppy = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}.get(timeframe, 8760)
    return df, compute_risk_report(df, periods_per_year=ppy)


def run_backtest_inverse_sma(symbol="AAPL", timeframe="1d", limit=1000, since=None, until=None, sma_fast=20, sma_slow=50, commission_pct=0.001, slippage_pct=0.0002, plot=True, verbose=True):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)
    df = generate_signals_inverse_sma(df, sma_fast, sma_slow)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct)
    ppy = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}.get(timeframe, 8760)
    return df, compute_risk_report(df, periods_per_year=ppy)


if __name__ == "__main__":
    df, report = run_backtest(
        symbol="AAPL",
        timeframe="1h",
        limit=1000,
        sma_fast=20,
        sma_slow=50,
        short_allowed=False,
    )

    print("\n" + "=" * 50)
    print("RISK REPORT")
    print("=" * 50)
    print(f"Sharpe Ratio       : {report['sharpe_ratio']:.2f}")
    print(f"Max Drawdown      : {report['max_drawdown_pct']:.2f}%")
    print(f"Win Rate          : {report['win_rate_pct']:.1f}%")
    print(f"Number of trades  : {report['n_trades']}")
    print(f"Strategy Return   : {report['total_return_pct']:.2f}%")
    print(f"Buy & Hold Return : {report['bh_return_pct']:.2f}%")
    print("=" * 50)
