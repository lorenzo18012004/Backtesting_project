"""
Portfolio backtesting : stratégies multi-actifs style hedge fund.
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import PERIODS_PER_YEAR
from .data import fetch_ohlcv, clean_ohlcv, compute_log_returns
from .pnl import compute_strategy_returns, apply_costs
from .risk import compute_risk_report
from .signals import generate_signals

logger = logging.getLogger(__name__)


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
    periods_per_year = PERIODS_PER_YEAR.get(timeframe, 8760)

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

    if not symbols:
        return pd.DataFrame(), {}

    common_idx = dfs[symbols[0]].index
    for sym in symbols[1:]:
        common_idx = common_idx.intersection(dfs[sym].index)
    common_idx = common_idx.sort_values()

    n = len(common_idx)
    if n == 0:
        return pd.DataFrame(), {}
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
    except Exception as e:
        logger.debug("Markowitz max Sharpe failed: %s", e)
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
    except Exception as e:
        logger.debug("Markowitz max IR failed: %s", e)
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
    periods_per_year = PERIODS_PER_YEAR.get(timeframe, 365)
    warmup = max(markowitz_lookback, factor_value_lb) + 100
    since_fetch = since
    until_fetch = until
    if since is not None and until is not None:
        tf_days = {"1d": 1, "1wk": 7, "1mo": 30}.get(timeframe, 1)
        since_fetch = since - timedelta(days=int(warmup * tf_days * 1.6))

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

    if not symbols:
        return pd.DataFrame(), {}

    common_idx = dfs[symbols[0]].index
    for sym in symbols[1:]:
        common_idx = common_idx.intersection(dfs[sym].index)
    common_idx = common_idx.sort_values()
    n = len(common_idx)

    if n == 0:
        return pd.DataFrame(), {}

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
    except Exception as e:
        logger.debug("S&P 500 fetch failed: %s", e)

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
                    except Exception as e:
                        logger.debug("Optimization trial failed: %s", e)

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
        except Exception as e:
            logger.debug("Objective eval failed: %s", e)
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
    except Exception as e:
        logger.debug("Portfolio optimization failed: %s", e)
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
