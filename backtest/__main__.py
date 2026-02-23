"""
Point d'entrée pour : python -m backtest
"""

from .core import run_backtest

if __name__ == "__main__":
    df, report = run_backtest(
        symbol="AAPL",
        timeframe="1d",
        limit=1000,
        sma_fast=20,
        sma_slow=50,
        short_allowed=False,
        plot=True,
        verbose=True,
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
