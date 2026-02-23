"""
Risk Report : métriques de risque (Sharpe, VaR, drawdown, etc.).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """Sharpe = E[R_p - R_f] / σ_p"""
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def max_drawdown(equity_curve: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Maximum Drawdown : plus grosse chute du capital."""
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
    """Pourcentage de trades gagnants."""
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
    """VaR historique : perte maximale attendue avec (1-confidence)% de probabilité."""
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """ES (Expected Shortfall / CVaR) : perte moyenne dans les pires (1-confidence)% des cas."""
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
    """Sortino = E[R - Rf] / σ_downside"""
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
    """Calmar = Rendement annualisé / Max Drawdown"""
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
    """Profit Factor = Somme des gains / |Somme des pertes|"""
    ret = df["strategy_return_net"]
    gains = ret[ret > 0].sum()
    losses = ret[ret < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return gains / abs(losses)


def monte_carlo_simulation(
    returns: pd.Series,
    n_simulations: int = 1000,
    periods_per_year: int = 8760,
    seed: Optional[int] = None,
) -> dict:
    """Monte Carlo : rééchantillonnage AVEC remise des rendements."""
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
    """Bootstrap : rééchantillonnage par blocs."""
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
    """Tests statistiques sur la distribution des rendements."""
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
    """Information Ratio = (R_strat - R_bench) / Tracking Error"""
    active = strategy_returns - benchmark_returns
    if active.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * active.mean() / active.std()


def rolling_metrics(
    df: pd.DataFrame,
    window: int = 252,
    periods_per_year: int = 8760,
) -> pd.DataFrame:
    """Métriques glissantes : Sharpe, Drawdown, Volatilité sur fenêtre roulante."""
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
    """Stress test : identifie les pires périodes."""
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
    """Rapport de risque complet : Sharpe, Sortino, Calmar, VaR, ES, Profit Factor, etc."""
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
    """Lance toutes les analyses pro : Monte Carlo, Bootstrap, tests statistiques, stress test, rolling metrics."""
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
