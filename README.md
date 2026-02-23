# 📈 Backtesting Engine

Système complet de backtesting pour stratégies de trading sur actions (Yahoo Finance). Projet portfolio — niveau Master Finance / Quantitative.

## Fonctionnalités

| Catégorie | Détails |
|-----------|---------|
| **Stratégies** | Buy & Hold, SMA Crossover + RSI, Portefeuille multi-actifs (style hedge fund) + Live |
| **Métriques de risque** | Sharpe, Sortino, Calmar, VaR, Expected Shortfall, Max Drawdown, Win Rate, Profit Factor |
| **Analyses avancées** | Monte Carlo, Bootstrap, tests statistiques (Jarque-Bera), stress test, rolling metrics |
| **Validation** | Walk-Forward (In-Sample / Out-of-Sample), optimisation des facteurs |

## Installation

```bash
git clone https://github.com/lorenzo18012004/Backtesting_project.git
cd Backtesting_project
pip install -r requirements.txt
```

## Utilisation

### Interface Streamlit (recommandé)

```bash
streamlit run app.py
```

### Ligne de commande (Python)

```bash
python -m backtest
```

### Utilisation en code

```python
from backtest import run_backtest, run_backtest_buy_hold, run_backtest_portfolio_hf

# SMA Crossover
df, report = run_backtest(
    symbol="AAPL",
    timeframe="1d",
    sma_fast=20,
    sma_slow=50,
    commission_pct=0.001,
    slippage_pct=0.0002,
)
print(f"Sharpe: {report['sharpe_ratio']:.2f}, Max DD: {report['max_drawdown_pct']:.1f}%")

# Buy & Hold
df, report = run_backtest_buy_hold(symbol="AAPL", timeframe="1d")

# Portefeuille multi-actifs
df, report = run_backtest_portfolio_hf(
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    timeframe="1d",
)
```

## Architecture

```
backtest/
├── config.py      # Constantes (timeframes, periods per year)
├── data.py        # Récupération OHLCV (Yahoo Finance), nettoyage, log-returns
├── signals.py     # Génération de signaux (SMA, Bollinger, MACD, RSI…)
├── pnl.py         # Rendements stratégie, frais, slippage
├── risk.py        # Métriques de risque (Sharpe, VaR, Monte Carlo…)
├── portfolio.py   # Stratégie multi-actifs (Markowitz, facteurs, VaR)
├── viz.py         # Graphiques (performance, drawdown)
└── core.py        # Orchestration (run_backtest, walk-forward)
```

### Pipeline (7 étapes)

1. **Data Pipeline** — Récupération OHLCV via Yahoo Finance
2. **Data Cleaning** — DataFrame Pandas, timestamps ISO8601, log-returns
3. **Strategy Layer** — Signaux avec shift(1) anti look-ahead
4. **PnL Engine** — Rendements stratégie, equity cumulée
5. **Reality Check** — Frais + slippage par trade
6. **Risk Report** — Sharpe, Max DD, VaR, etc.
7. **Visualizer** — Courbes performance et drawdown

## Tests

```bash
pytest tests/ -v
```

29 tests unitaires et d'intégration (signals, pnl, risk, fetch mocké, pipeline complet).

## Stack technique

- **Python 3.10+**
- **Pandas, NumPy, SciPy** — calculs et optimisation (Markowitz)
- **yfinance** — données OHLCV (actions, ETFs)
- **Streamlit** — interface web
- **Matplotlib** — visualisation
- **pytest** — tests

## Données

Les données proviennent de **Yahoo Finance** (gratuit, sans API key). Timeframes supportés : `1d`, `1wk`, `1mo`.
