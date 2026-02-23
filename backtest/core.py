"""
Core : orchestration des backtests et walk-forward.
"""

from datetime import datetime, timedelta
from typing import Tuple, Optional

import pandas as pd

from .config import PERIODS_PER_YEAR
from .data import fetch_ohlcv, clean_ohlcv, compute_log_returns
from .signals import (
    generate_signals,
    generate_signals_bollinger,
    generate_signals_buy_hold,
    generate_signals_macd,
    generate_signals_rsi_naive,
    generate_signals_inverse_sma,
)
from .pnl import compute_strategy_returns, apply_costs
from .risk import compute_risk_report
from .viz import get_price_chart_figure, get_plot_figure, plot_results
from . import portfolio


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
    """Récupère les données via Yahoo Finance et calcule les signaux actuels."""
    until = datetime.now()
    since = until - timedelta(days=limit * 2)
    results = {}
    for sym in symbols:
        try:
            ohlcv = fetch_ohlcv(sym, timeframe, limit=limit, since=since, until=until)
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
        except Exception:
            results[sym] = {"error": "Failed to fetch"}
    return results


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
    """Lance le backtest sur un DataFrame déjà nettoyé (pour walk-forward)."""
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
    """Lance le backtest Bollinger Bands sur un DataFrame déjà nettoyé."""
    df = df_raw.copy()
    df = compute_log_returns(df)
    df = generate_signals_bollinger(
        df, bb_period=bb_period, bb_std=bb_std, rsi_period=rsi_period,
        rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought,
        use_volume_filter=use_volume_filter, volume_ma_period=volume_ma_period,
        short_allowed=short_allowed, start_in_cash=start_in_cash,
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
    """Walk-Forward : split In-Sample (optimisation) / Out-of-Sample (validation)."""
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
            f"Available data: {data_min.date()} to {data_max.date()}."
        )
    split_in_valid = min(int(n_valid * in_sample_pct), n_valid - 1)
    split_pos = df.index.get_loc(valid_indices[split_in_valid])

    df_in = df.iloc[: split_pos + 1]
    df_out_raw = df.iloc[max(0, split_pos + 1 - warmup_bars):] if warmup_bars > 0 else df.iloc[split_pos + 1:]

    periods_per_year = PERIODS_PER_YEAR.get(timeframe, 8760)

    df_in_bt, report_in = run_backtest_on_df(
        df_in, sma_fast=sma_fast, sma_slow=sma_slow, short_allowed=short_allowed,
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

    oos_start_ts = valid_indices[split_in_valid] if split_in_valid < n_valid else df.index[split_pos]
    df_out_bt = df_out_bt[df_out_bt.index >= oos_start_ts] if len(df_out_bt) > 0 else df_out_bt
    report_out = compute_risk_report(df_out_bt, periods_per_year=periods_per_year) if len(df_out_bt) > 0 else report_out

    return {
        "in_sample": {"df": df_in_bt, "report": report_in, "period": (df_in.index[0], df_in.index[-1])},
        "out_of_sample": {"df": df_out_bt, "report": report_out, "period": (oos_start_ts, df.index[-1]) if len(df_out_bt) > 0 else (None, None)},
        "robust": report_in["total_return_pct"] > 0 and report_out["total_return_pct"] > 0,
    }


def run_walk_forward_backtest_bollinger(
    symbol: str, timeframe: str, since: datetime, until: datetime,
    in_sample_pct: float = 0.6, bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0,
    short_allowed: bool = False, start_in_cash: bool = True,
    use_volume_filter: bool = True, volume_ma_period: int = 20,
    commission_pct: float = 0.001, slippage_pct: float = 0.0002,
    stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None,
) -> dict:
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    split_idx = int(len(df) * in_sample_pct)
    df_in = df.iloc[:split_idx]
    df_out = df.iloc[split_idx:]
    periods_per_year = PERIODS_PER_YEAR.get(timeframe, 8760)
    df_in_bt, report_in = run_backtest_on_df_bollinger(df_in, bb_period, bb_std, rsi_period, rsi_oversold, rsi_overbought, short_allowed, start_in_cash, use_volume_filter, volume_ma_period, commission_pct, slippage_pct, stop_loss_pct, take_profit_pct, periods_per_year)
    df_out_bt, report_out = run_backtest_on_df_bollinger(df_out, bb_period, bb_std, rsi_period, rsi_oversold, rsi_overbought, short_allowed, start_in_cash, use_volume_filter, volume_ma_period, commission_pct, slippage_pct, stop_loss_pct, take_profit_pct, periods_per_year)
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
    ppy = PERIODS_PER_YEAR.get(timeframe, 8760)
    df_in_bt, r_in = run_backtest_on_df_buy_hold(df.iloc[:split_idx], commission_pct, slippage_pct, periods_per_year=ppy, spread_pct=spread_pct)
    df_out_bt, r_out = run_backtest_on_df_buy_hold(df.iloc[split_idx:], commission_pct, slippage_pct, periods_per_year=ppy, spread_pct=spread_pct)
    return {"in_sample": {"df": df_in_bt, "report": r_in, "period": (df.index[0], df.iloc[split_idx - 1].name)},
            "out_of_sample": {"df": df_out_bt, "report": r_out, "period": (df.iloc[split_idx].name, df.index[-1])},
            "robust": r_in["total_return_pct"] > 0 and r_out["total_return_pct"] > 0}


def run_walk_forward_backtest_macd(symbol, timeframe, since, until, in_sample_pct=0.6, macd_fast=12, macd_slow=26, macd_signal=9, commission_pct=0.001, slippage_pct=0.0002):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    split_idx = int(len(df) * in_sample_pct)
    ppy = PERIODS_PER_YEAR.get(timeframe, 8760)
    df_in_bt, r_in = run_backtest_on_df_macd(df.iloc[:split_idx], macd_fast, macd_slow, macd_signal, commission_pct, slippage_pct, ppy)
    df_out_bt, r_out = run_backtest_on_df_macd(df.iloc[split_idx:], macd_fast, macd_slow, macd_signal, commission_pct, slippage_pct, ppy)
    return {"in_sample": {"df": df_in_bt, "report": r_in, "period": (df.index[0], df.iloc[split_idx - 1].name)},
            "out_of_sample": {"df": df_out_bt, "report": r_out, "period": (df.iloc[split_idx].name, df.index[-1])},
            "robust": r_in["total_return_pct"] > 0 and r_out["total_return_pct"] > 0}


def run_walk_forward_backtest_rsi_naive(symbol, timeframe, since, until, in_sample_pct=0.6, rsi_period=14, rsi_oversold=30, rsi_overbought=70, commission_pct=0.001, slippage_pct=0.0002):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    split_idx = int(len(df) * in_sample_pct)
    ppy = PERIODS_PER_YEAR.get(timeframe, 8760)
    df_in_bt, r_in = run_backtest_on_df_rsi_naive(df.iloc[:split_idx], rsi_period, rsi_oversold, rsi_overbought, commission_pct, slippage_pct, ppy)
    df_out_bt, r_out = run_backtest_on_df_rsi_naive(df.iloc[split_idx:], rsi_period, rsi_oversold, rsi_overbought, commission_pct, slippage_pct, ppy)
    return {"in_sample": {"df": df_in_bt, "report": r_in, "period": (df.index[0], df.iloc[split_idx - 1].name)},
            "out_of_sample": {"df": df_out_bt, "report": r_out, "period": (df.iloc[split_idx].name, df.index[-1])},
            "robust": r_in["total_return_pct"] > 0 and r_out["total_return_pct"] > 0}


def run_walk_forward_backtest_inverse_sma(symbol, timeframe, since, until, in_sample_pct=0.6, sma_fast=20, sma_slow=50, commission_pct=0.001, slippage_pct=0.0002):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    split_idx = int(len(df) * in_sample_pct)
    ppy = PERIODS_PER_YEAR.get(timeframe, 8760)
    df_in_bt, r_in = run_backtest_on_df_inverse_sma(df.iloc[:split_idx], sma_fast, sma_slow, commission_pct, slippage_pct, ppy)
    df_out_bt, r_out = run_backtest_on_df_inverse_sma(df.iloc[split_idx:], sma_fast, sma_slow, commission_pct, slippage_pct, ppy)
    return {"in_sample": {"df": df_in_bt, "report": r_in, "period": (df.index[0], df.iloc[split_idx - 1].name)},
            "out_of_sample": {"df": df_out_bt, "report": r_out, "period": (df.iloc[split_idx].name, df.index[-1])},
            "robust": r_in["total_return_pct"] > 0 and r_out["total_return_pct"] > 0}


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
    """Exécute le backtest complet et retourne (DataFrame, rapport de risque)."""
    def _log(msg):
        if verbose:
            try:
                print(msg)
            except (ValueError, OSError):
                pass

    _log("Étape 1 : Récupération des données...")
    warmup_bars = max(sma_slow, rsi_period if use_rsi_filter else 0, volume_ma_period if use_volume_filter else 0) if (since is not None and until is not None) else 0
    since_fetch, until_fetch = since, until
    if since is not None and until is not None and warmup_bars > 0:
        tf_days = {"1d": 1, "1wk": 7, "1mo": 30}.get(timeframe, 1)
        since_fetch = since - timedelta(days=warmup_bars * tf_days)
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since_fetch, until=until_fetch, data_source=data_source)

    _log("Étape 2 : Nettoyage des données...")
    df = clean_ohlcv(ohlcv_raw)
    if df.empty or len(df) == 0:
        raise ValueError(f"No data for {symbol}. Check ticker and period.")
    df = compute_log_returns(df)

    _log("Étape 3 : Génération des signaux...")
    df = generate_signals(df, sma_fast=sma_fast, sma_slow=sma_slow, short_allowed=short_allowed, start_in_cash=start_in_cash,
        use_rsi_filter=use_rsi_filter, rsi_period=rsi_period, rsi_long_max=rsi_long_max, rsi_short_min=rsi_short_min,
        use_volume_filter=use_volume_filter, volume_ma_period=volume_ma_period)

    _log("Étape 4 : Calcul des performances...")
    df = compute_strategy_returns(df, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)

    _log("Étape 5 : Application des frais et slippage...")
    df = apply_costs(df, commission_pct, slippage_pct, spread_pct)

    if since is not None and until is not None and warmup_bars > 0 and len(df) > 0:
        since_ts, until_ts = pd.Timestamp(since), pd.Timestamp(until)
        mask = (df.index >= since_ts) & (df.index <= until_ts)
        df = df.loc[mask]

    _log("Étape 6 : Rapport de risque...")
    periods_per_year = PERIODS_PER_YEAR.get(timeframe, 8760)
    report = compute_risk_report(df, periods_per_year=periods_per_year)

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
    def _log(msg):
        if verbose:
            try:
                print(msg)
            except (ValueError, OSError):
                pass

    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)
    df = generate_signals_bollinger(df, bb_period=bb_period, bb_std=bb_std, rsi_period=rsi_period, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought, use_volume_filter=use_volume_filter, volume_ma_period=volume_ma_period, short_allowed=short_allowed, start_in_cash=start_in_cash)
    df = compute_strategy_returns(df, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
    df = apply_costs(df, commission_pct, slippage_pct)
    periods_per_year = PERIODS_PER_YEAR.get(timeframe, 8760)
    report = compute_risk_report(df, periods_per_year=periods_per_year)

    if plot:
        import matplotlib.pyplot as plt
        fig = get_price_chart_figure(df, title=f"{symbol} | Bollinger BB{bb_period}", strategy_type="bollinger")
        plt.show()

    return df, report


def run_backtest_buy_hold(symbol="AAPL", timeframe="1d", limit=1000, since=None, until=None, commission_pct=0.001, slippage_pct=0.0002, spread_pct=0.0, data_source="yahoo", plot=True, verbose=True):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until, data_source=data_source)
    df = clean_ohlcv(ohlcv_raw)
    if df.empty or len(df) == 0:
        raise ValueError(f"No data for {symbol}. Check ticker and period.")
    df = compute_log_returns(df)
    df = generate_signals_buy_hold(df)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct, spread_pct)
    ppy = PERIODS_PER_YEAR.get(timeframe, 8760)
    return df, compute_risk_report(df, periods_per_year=ppy)


def run_backtest_macd(symbol="AAPL", timeframe="1d", limit=1000, since=None, until=None, macd_fast=12, macd_slow=26, macd_signal=9, commission_pct=0.001, slippage_pct=0.0002, plot=True, verbose=True):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)
    df = generate_signals_macd(df, fast=macd_fast, slow=macd_slow, signal_period=macd_signal)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct)
    ppy = PERIODS_PER_YEAR.get(timeframe, 8760)
    return df, compute_risk_report(df, periods_per_year=ppy)


def run_backtest_rsi_naive(symbol="AAPL", timeframe="1d", limit=1000, since=None, until=None, rsi_period=14, rsi_oversold=30, rsi_overbought=70, commission_pct=0.001, slippage_pct=0.0002, plot=True, verbose=True):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)
    df = generate_signals_rsi_naive(df, rsi_period, rsi_oversold, rsi_overbought)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct)
    ppy = PERIODS_PER_YEAR.get(timeframe, 8760)
    return df, compute_risk_report(df, periods_per_year=ppy)


def run_backtest_inverse_sma(symbol="AAPL", timeframe="1d", limit=1000, since=None, until=None, sma_fast=20, sma_slow=50, commission_pct=0.001, slippage_pct=0.0002, plot=True, verbose=True):
    ohlcv_raw = fetch_ohlcv(symbol, timeframe, limit=limit, since=since, until=until)
    df = clean_ohlcv(ohlcv_raw)
    df = compute_log_returns(df)
    df = generate_signals_inverse_sma(df, sma_fast, sma_slow)
    df = compute_strategy_returns(df)
    df = apply_costs(df, commission_pct, slippage_pct)
    ppy = PERIODS_PER_YEAR.get(timeframe, 8760)
    return df, compute_risk_report(df, periods_per_year=ppy)
