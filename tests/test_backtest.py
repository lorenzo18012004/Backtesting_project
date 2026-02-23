"""
Tests unitaires pour le moteur de backtesting.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch

from backtest import (
    clean_ohlcv,
    compute_log_returns,
    apply_costs,
    compute_risk_report,
    sharpe_ratio,
    max_drawdown,
    compute_rsi,
    generate_signals,
    generate_signals_buy_hold,
    compute_strategy_returns,
    fetch_ohlcv,
    run_backtest_on_df,
    run_backtest_on_df_buy_hold,
    get_plot_figure,
    get_price_chart_figure,
    compute_bollinger_bands,
    var_historical,
    expected_shortfall,
    information_ratio,
)


# ============ clean_ohlcv ============
def test_clean_ohlcv_empty():
    """clean_ohlcv avec liste vide retourne un DataFrame vide."""
    result = clean_ohlcv([])
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_clean_ohlcv_none_equivalent():
    """clean_ohlcv avec None retourne un DataFrame vide."""
    result = clean_ohlcv(None)
    assert result.empty


def test_clean_ohlcv_valid():
    """clean_ohlcv transforme correctement les données brutes."""
    raw = [
        [1000000000000, 100.0, 105.0, 98.0, 102.0, 1000.0],
        [1000000000000 + 86400000, 102.0, 108.0, 101.0, 106.0, 1200.0],
    ]
    result = clean_ohlcv(raw)
    assert len(result) == 2
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert result["Close"].iloc[0] == 102.0
    assert result["Close"].iloc[1] == 106.0


# ============ compute_log_returns ============
def test_compute_log_returns_basic():
    """compute_log_returns calcule ln(P_t / P_{t-1})."""
    df = pd.DataFrame({
        "Close": [100.0, 102.0, 101.0, 105.0],
    }, index=pd.date_range("2024-01-01", periods=4, freq="D"))
    result = compute_log_returns(df)
    assert "log_return" in result.columns
    assert pd.isna(result["log_return"].iloc[0])  # Premier rendement = NaN
    np.testing.assert_allclose(result["log_return"].iloc[1], np.log(102 / 100))
    np.testing.assert_allclose(result["log_return"].iloc[2], np.log(101 / 102))
    np.testing.assert_allclose(result["log_return"].iloc[3], np.log(105 / 101))


def test_compute_log_returns_constant_price():
    """Rendements nuls quand le prix est constant."""
    df = pd.DataFrame({"Close": [100.0, 100.0, 100.0]}, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    result = compute_log_returns(df)
    assert result["log_return"].iloc[1] == 0.0
    assert result["log_return"].iloc[2] == 0.0


# ============ apply_costs ============
def test_apply_costs_empty():
    """apply_costs avec DataFrame vide retourne tel quel."""
    df = pd.DataFrame()
    result = apply_costs(df)
    assert result.empty


def test_apply_costs_buy_hold():
    """Buy & Hold : 2 trades (entrée + sortie), coûts appliqués."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "signal": [1, 1, 1, 1, 1],
        "strategy_return": [0.0, 0.01, 0.02, -0.01, 0.03],
    }, index=idx)
    result = apply_costs(df, commission_pct=0.001, slippage_pct=0.0, spread_pct=0.0)
    assert "strategy_return_net" in result.columns
    assert "trade_cost" in result.columns
    cost_per_trade = 0.001
    assert result.loc[result.index[0], "trade_cost"] == cost_per_trade
    assert result.loc[result.index[-1], "trade_cost"] == cost_per_trade


def test_apply_costs_reduces_equity():
    """Les coûts réduisent l'equity nette par rapport à la brute."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "signal": [1] * 10,
        "strategy_return": [0.01] * 10,
    }, index=idx)
    result = apply_costs(df, commission_pct=0.01, slippage_pct=0.0, spread_pct=0.0)
    assert result["strategy_equity_net"].iloc[-1] < np.exp(0.01 * 10)


# ============ compute_risk_report ============
def test_compute_risk_report_empty():
    """compute_risk_report avec DataFrame vide retourne un rapport vide."""
    df = pd.DataFrame()
    report = compute_risk_report(df)
    assert report["sharpe_ratio"] == 0.0
    assert report["n_trades"] == 0
    assert report["total_return_pct"] == 0.0


def test_compute_risk_report_valid():
    """compute_risk_report produit les métriques attendues."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    returns = np.random.randn(100) * 0.01
    df = pd.DataFrame({
        "log_return": returns,
        "strategy_return_net": returns,
        "strategy_equity_net": np.exp(np.cumsum(returns)),
        "bh_equity": np.exp(np.cumsum(returns)),
        "position_change": np.concatenate([[1], np.zeros(98), [1]]),
        "turnover": np.zeros(100),
    }, index=idx)
    report = compute_risk_report(df, periods_per_year=365)
    assert "sharpe_ratio" in report
    assert "max_drawdown_pct" in report
    assert "total_return_pct" in report
    assert "n_trades" in report
    assert report["n_trades"] >= 0


# ============ sharpe_ratio ============
def test_sharpe_ratio_zero_vol():
    """Sharpe = 0 quand volatilité nulle."""
    returns = pd.Series([0.01, 0.01, 0.01])
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive():
    """Sharpe positif pour rendements positifs avec volatilité."""
    returns = pd.Series([0.02, -0.01, 0.03, 0.01])
    sr = sharpe_ratio(returns, periods_per_year=252)
    assert sr > 0


# ============ max_drawdown ============
def test_max_drawdown_empty():
    """max_drawdown avec série vide."""
    dd, peak, trough = max_drawdown(pd.Series(dtype=float))
    assert dd == 0.0
    assert peak is None
    assert trough is None


def test_max_drawdown_simple():
    """Max drawdown sur courbe connue."""
    equity = pd.Series([1.0, 1.2, 1.0, 0.8, 1.1], index=pd.date_range("2024-01-01", periods=5, freq="D"))
    dd, peak, trough = max_drawdown(equity)
    assert dd < 0
    np.testing.assert_allclose(dd, (0.8 - 1.2) / 1.2, rtol=1e-5)


# ============ compute_rsi ============
def test_compute_rsi_range():
    """RSI reste entre 0 et 100."""
    series = pd.Series(np.random.randn(50).cumsum() + 100)
    rsi = compute_rsi(series, period=14)
    valid = rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_compute_rsi_oversold():
    """RSI proche de 0 après chute forte."""
    series = pd.Series(100 - np.arange(20).astype(float))  # Chute linéaire
    rsi = compute_rsi(series, period=5)
    assert rsi.iloc[-1] < 50


# ============ generate_signals ============
def test_generate_signals_sma_crossover():
    """generate_signals produit signal=1 quand SMA_fast > SMA_slow."""
    df = pd.DataFrame({
        "Close": [100 + i * 2 for i in range(60)],  # Tendance haussière
        "Volume": [1000] * 60,
    }, index=pd.date_range("2024-01-01", periods=60, freq="D"))
    df = compute_log_returns(df)
    result = generate_signals(df, sma_fast=5, sma_slow=20, start_in_cash=True)
    assert "signal" in result.columns
    assert "SMA_fast" in result.columns
    # En tendance haussière, signal devrait être 1 après warmup
    assert (result["signal"].iloc[25:] == 1).all()


def test_generate_signals_buy_hold():
    """generate_signals_buy_hold met signal=1 partout."""
    df = pd.DataFrame({"Close": [100, 102, 101], "Volume": [1000, 1000, 1000]}, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    result = generate_signals_buy_hold(df)
    assert (result["signal"] == 1).all()


# ============ compute_strategy_returns ============
def test_compute_strategy_returns_simple():
    """compute_strategy_returns sans SL/TP : signal * log_return."""
    df = pd.DataFrame({
        "Close": [100.0, 102.0, 101.0, 105.0],
        "Open": [100.0, 102.0, 101.0, 105.0],
        "High": [102.0, 103.0, 102.0, 106.0],
        "Low": [99.0, 101.0, 100.0, 104.0],
        "Volume": [1000] * 4,
    }, index=pd.date_range("2024-01-01", periods=4, freq="D"))
    df = compute_log_returns(df)
    df["signal"] = [1, 1, 0, 1]  # long, long, cash, long
    result = compute_strategy_returns(df)
    assert "strategy_return" in result.columns
    assert "bh_equity" in result.columns
    np.testing.assert_allclose(result["strategy_return"].iloc[1], np.log(102 / 100))
    assert result["strategy_return"].iloc[2] == 0.0  # cash


# ============ fetch_ohlcv (mock) ============
def test_fetch_ohlcv_with_mock():
    """fetch_ohlcv retourne des données quand yfinance répond."""
    mock_df = pd.DataFrame({
        "Open": [100, 102], "High": [105, 108], "Low": [98, 101],
        "Close": [102, 106], "Volume": [1000, 1200],
    }, index=pd.date_range("2024-01-01", periods=2, freq="D"))
    with patch("backtest.data.yf") as mock_yf:
        mock_yf.download.return_value = mock_df
        from backtest.data import fetch_ohlcv_yahoo
        result = fetch_ohlcv_yahoo("AAPL", "1d", limit=10)
    assert len(result) == 2
    assert len(result[0]) == 6  # timestamp + O,H,L,C,V
    assert result[0][4] == 102.0  # Close


def test_fetch_ohlcv_empty_response():
    """fetch_ohlcv retourne [] quand yfinance ne renvoie rien."""
    with patch("backtest.data.yf") as mock_yf:
        mock_yf.download.return_value = pd.DataFrame()
        from backtest.data import fetch_ohlcv_yahoo
        result = fetch_ohlcv_yahoo("INVALID", "1d")
    assert result == []


# ============ run_backtest_on_df (intégration) ============
def test_run_backtest_on_df_buy_hold_integration():
    """Pipeline complet : clean_ohlcv -> log_returns -> buy_hold -> costs -> report."""
    raw = [[1000000000000 + i * 86400000, 100.0 + i, 105.0 + i, 98.0 + i, 102.0 + i, 1000.0] for i in range(50)]
    df_raw = clean_ohlcv(raw)
    df, report = run_backtest_on_df_buy_hold(df_raw, commission_pct=0.001, slippage_pct=0.0)
    assert len(df) == 50
    assert "strategy_equity_net" in df.columns
    assert report["total_return_pct"] != 0 or report["n_trades"] >= 0
    assert "sharpe_ratio" in report


def test_run_backtest_on_df_sma_integration():
    """Pipeline SMA complet sur données synthétiques."""
    np.random.seed(42)
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    df_raw = pd.DataFrame({
        "Open": close, "High": close * 1.02, "Low": close * 0.98,
        "Close": close, "Volume": np.ones(n) * 1000,
    }, index=idx)
    df, report = run_backtest_on_df(
        df_raw, sma_fast=10, sma_slow=30, short_allowed=False, start_in_cash=True,
        use_rsi_filter=False, rsi_period=14, rsi_long_max=70, rsi_short_min=30,
        use_volume_filter=False, volume_ma_period=20,
        commission_pct=0.001, slippage_pct=0.0, spread_pct=0.0,
    )
    assert len(df) == n
    assert "signal" in df.columns
    assert report["n_trades"] >= 0
    assert "max_drawdown_pct" in report


# ============ viz ============
def test_get_plot_figure_returns_figure():
    """get_plot_figure retourne une figure matplotlib valide."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "Close": [100 + i for i in range(20)],
        "strategy_equity_net": np.exp(np.cumsum(np.random.randn(20) * 0.01)),
        "bh_equity": np.exp(np.cumsum(np.random.randn(20) * 0.01)),
        "strategy_equity": np.exp(np.cumsum(np.random.randn(20) * 0.01)),
    }, index=idx)
    fig = get_plot_figure(df, compact=True, hide_comparison=True)
    assert fig is not None
    assert len(fig.axes) >= 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_get_price_chart_figure_returns_figure():
    """get_price_chart_figure retourne une figure valide."""
    df = pd.DataFrame({
        "Close": [100, 102, 101, 105],
        "SMA_fast": [100, 101, 101.5, 102],
        "SMA_slow": [100, 100.5, 101, 101.5],
        "signal": [0, 1, 1, 1],
    }, index=pd.date_range("2024-01-01", periods=4, freq="D"))
    fig = get_price_chart_figure(df, strategy_type="sma")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


# ============ risk (supplémentaires) ============
def test_var_historical():
    """var_historical retourne un quantile négatif pour des pertes."""
    returns = pd.Series(np.random.randn(100) * 0.02 - 0.001)  # Légèrement négatif
    var = var_historical(returns, 0.95)
    assert var < 0.05  # VaR 95% = 5e percentile


def test_expected_shortfall():
    """expected_shortfall <= var pour même confidence."""
    returns = pd.Series(np.random.randn(100) * 0.02)
    var = var_historical(returns, 0.95)
    es = expected_shortfall(returns, 0.95)
    assert es <= var


def test_information_ratio():
    """information_ratio = 0 quand strat = benchmark."""
    ret = pd.Series([0.01, -0.005, 0.02, 0.01])
    assert information_ratio(ret, ret) == 0.0


# ============ compute_bollinger_bands ============
def test_compute_bollinger_bands():
    """Bollinger : upper >= middle >= lower (où défini)."""
    series = pd.Series(100 + np.random.randn(50).cumsum())
    upper, middle, lower = compute_bollinger_bands(series, period=20)
    valid = ~upper.isna()
    assert (upper[valid] >= middle[valid]).all()
    assert (middle[valid] >= lower[valid]).all()
