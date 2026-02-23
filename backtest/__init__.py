"""
Backtesting Engine - Système complet de backtesting
Étapes 1 à 7 : Data Pipeline → Data Cleaning → Strategy → PnL → Reality Check → Risk Report → Visualizer
"""

import sys
import io

# Fix encoding Windows pour les accents (sauf sous Streamlit)
if sys.platform == "win32" and "streamlit" not in sys.modules:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Re-exports pour compatibilité : from backtest import X
from .config import TF_SECONDS, PERIODS_PER_YEAR
from .data import fetch_ohlcv, fetch_ohlcv_yahoo, clean_ohlcv, compute_log_returns
from .signals import (
    compute_rsi,
    generate_signals,
    compute_bollinger_bands,
    generate_signals_bollinger,
    generate_signals_buy_hold,
    compute_macd,
    generate_signals_macd,
    generate_signals_rsi_naive,
    generate_signals_inverse_sma,
)
from .pnl import compute_strategy_returns, apply_costs
from .risk import (
    sharpe_ratio,
    max_drawdown,
    win_rate,
    var_historical,
    expected_shortfall,
    sortino_ratio,
    calmar_ratio,
    profit_factor,
    monte_carlo_simulation,
    bootstrap_analysis,
    statistical_tests,
    information_ratio,
    rolling_metrics,
    stress_test,
    compute_risk_report,
    run_pro_analysis,
)
from .viz import get_price_chart_figure, get_plot_figure, plot_results
from . import portfolio
from .core import (
    get_live_signals,
    run_backtest_on_df,
    run_backtest_on_df_bollinger,
    run_backtest_on_df_buy_hold,
    run_backtest_on_df_macd,
    run_backtest_on_df_rsi_naive,
    run_backtest_on_df_inverse_sma,
    run_walk_forward_backtest,
    run_walk_forward_backtest_bollinger,
    run_walk_forward_backtest_buy_hold,
    run_walk_forward_backtest_macd,
    run_walk_forward_backtest_rsi_naive,
    run_walk_forward_backtest_inverse_sma,
    run_backtest,
    run_backtest_bollinger,
    run_backtest_buy_hold,
    run_backtest_macd,
    run_backtest_rsi_naive,
    run_backtest_inverse_sma,
)

# Portfolio HF
run_backtest_portfolio_hf = portfolio.run_backtest_portfolio_hf
run_walk_forward_backtest_portfolio_hf = portfolio.run_walk_forward_backtest_portfolio_hf
run_backtest_portfolio = portfolio.run_backtest_portfolio

__all__ = [
    "TF_SECONDS", "PERIODS_PER_YEAR",
    "fetch_ohlcv", "fetch_ohlcv_yahoo", "clean_ohlcv", "compute_log_returns",
    "compute_rsi", "generate_signals", "compute_bollinger_bands", "generate_signals_bollinger",
    "generate_signals_buy_hold", "compute_macd", "generate_signals_macd",
    "generate_signals_rsi_naive", "generate_signals_inverse_sma",
    "compute_strategy_returns", "apply_costs",
    "sharpe_ratio", "max_drawdown", "win_rate", "var_historical", "expected_shortfall",
    "sortino_ratio", "calmar_ratio", "profit_factor",
    "monte_carlo_simulation", "bootstrap_analysis", "statistical_tests",
    "information_ratio", "rolling_metrics", "stress_test",
    "compute_risk_report", "run_pro_analysis",
    "get_price_chart_figure", "get_plot_figure", "plot_results",
    "get_live_signals",
    "run_backtest_on_df", "run_backtest_on_df_bollinger", "run_backtest_on_df_buy_hold",
    "run_backtest_on_df_macd", "run_backtest_on_df_rsi_naive", "run_backtest_on_df_inverse_sma",
    "run_walk_forward_backtest", "run_walk_forward_backtest_bollinger", "run_walk_forward_backtest_buy_hold",
    "run_walk_forward_backtest_macd", "run_walk_forward_backtest_rsi_naive", "run_walk_forward_backtest_inverse_sma",
    "run_backtest", "run_backtest_bollinger", "run_backtest_buy_hold",
    "run_backtest_macd", "run_backtest_rsi_naive", "run_backtest_inverse_sma",
    "run_backtest_portfolio", "run_backtest_portfolio_hf", "run_walk_forward_backtest_portfolio_hf",
]
