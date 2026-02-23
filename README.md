# Backtesting Engine - Actions

Système complet de backtesting pour stratégies de trading actions, niveau Master Finance.

## Architecture (7 étapes)

| Étape | Module | Description |
|-------|--------|-------------|
| 1 | **Data Pipeline** | Récupération OHLCV via Yahoo Finance (actions, ETFs) |
| 2 | **Data Cleaning** | DataFrame Pandas, timestamps ISO8601, log-returns |
| 3 | **Strategy Layer** | SMA 20/50 crossover, shift(1) anti look-ahead |
| 4 | **PnL Engine** | Rendements stratégie, equity cumulée |
| 5 | **Reality Check** | Frais 0,10% + slippage 0,02% par trade |
| 6 | **Risk Report** | Sharpe, Max Drawdown, Win Rate |
| 7 | **Visualizer** | Buy & Hold vs Strategy + courbe de Drawdown |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Interface Streamlit (recommandé)

```bash
streamlit run app.py
```

### Script en ligne de commande

```bash
python backtest.py
```

### Paramètres personnalisables

```python
df, report = run_backtest(
    symbol="AAPL",          # Action
    timeframe="1d",         # 1d, 1wk, 1mo
    limit=1000,             # Nombre de bougies
    sma_fast=20,
    sma_slow=50,
    short_allowed=False,    # True pour vente à découvert
    commission_pct=0.001,   # 0,10% (broker actions)
    slippage_pct=0.0002,    # 0,02%
    plot=True,
)
```

## Formules clés

- **Log-returns** : \( r_t = \ln(P_t / P_{t-1}) \)
- **Sharpe Ratio** : \( \frac{E[R_p - R_f]}{\sigma_p} \)
- **Signal** : SMA_fast > SMA_slow → 1 (long), sinon 0 ou -1 (short)

## Fichiers générés

- `backtest_results.png` : Graphique performance + drawdown
