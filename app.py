"""
Backtesting Engine - Multi-stratégies
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import html
matplotlib.use("Agg")
from datetime import datetime, timedelta

from backtest import (
    run_backtest,
    run_backtest_buy_hold,
    run_backtest_portfolio_hf,
    run_walk_forward_backtest,
    run_walk_forward_backtest_buy_hold,
    run_walk_forward_backtest_portfolio_hf,
    run_pro_analysis,
    compute_risk_report,
    get_plot_figure,
)

st.set_page_config(page_title="Backtesting", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ============ STYLE ============
st.markdown("""
<style>
    /* Thème sombre cohérent */
    .stApp { background: #0f172a; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1222 0%, #1e293b 100%);
        border-right: 1px solid rgba(148,163,184,0.15);
    }
    div[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    
    /* Header principal */
    .main-header { font-size: 2rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
    .main-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; }
    
    /* Cartes stratégie - grille 2x2 */
    .strategy-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.95) 100%);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(148,163,184,0.15);
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .strategy-card:hover {
        border-color: rgba(59,130,246,0.5);
        box-shadow: 0 8px 28px rgba(59,130,246,0.15);
        transform: translateY(-2px);
    }
    .strategy-card h3 {
        font-size: 1.35rem;
        font-weight: 600;
        color: #f8fafc;
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .strategy-card .card-desc {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0.5rem 0;
    }
    .strategy-card .card-detail {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.45;
        margin: 0.25rem 0 1rem 0;
    }
    
    /* Métriques */
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem; }
    
    /* Boutons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: none;
    }
    
    /* Sections */
    .section-title { font-size: 1.15rem; font-weight: 600; color: #e2e8f0; margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(148,163,184,0.2); }
    .success-box { padding: 1rem 1.25rem; border-radius: 10px; background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); }
    .info-box { padding: 1rem 1.25rem; border-radius: 10px; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.25); }
</style>
""", unsafe_allow_html=True)

# ============ CONFIG STRATÉGIES ============
STRATEGIES = {
    "buy_hold": {
        "name": "Buy & Hold",
        "description": "Toujours en position long, aucun trading actif.",
        "detail": "Référence pour comparer. Performe bien en bull market.",
        "icon": "🏠",
    },
    "sma_crossover_rsi": {
        "name": "SMA Crossover + RSI",
        "description": "Croisement SMA rapide/lente avec filtre RSI.",
        "detail": "Achat au croisement haussier. N'achète pas si RSI > 70 (surchauffe).",
        "icon": "📈",
    },
    "portfolio_hf": {
        "name": "Backtesting Portfolio",
        "description": "Portefeuille multi-actifs : Momentum, Trend, Low Vol.",
        "detail": "Optimisation Markowitz, contraintes de risque (VaR, corrélation, circuit breaker).",
        "icon": "🏛️",
    },
    "live": {
        "name": "Live",
        "description": "Backtesting Portfolio en live.",
        "detail": "Même stratégie que Backtesting Portfolio, lancée le 23/02/2026. Capital 10 000 €. PnL et évolution à jour à chaque visite.",
        "icon": "🔴",
    },
}

# ============ SESSION STATE ============
if "selected_strategy" not in st.session_state:
    st.session_state.selected_strategy = None

# ============ PAGE D'ACCUEIL ============
if st.session_state.selected_strategy is None:
    st.markdown('<p class="main-header">📈 Backtesting</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-sub">Choisis une stratégie pour lancer le backtest</p>', unsafe_allow_html=True)

    strats_list = list(STRATEGIES.items())
    c1, c2 = st.columns(2)
    def _card_html(s):
        desc = html.escape(s["description"])
        detail = html.escape(s.get("detail", ""))
        return f'''
        <div class="strategy-card">
            <h3>{s["icon"]} {s["name"]}</h3>
            <p class="card-desc">{desc}</p>
            <p class="card-detail">{detail}</p>
        </div>
        '''
    with c1:
        for j in [0, 1]:
            strat_id, strat = strats_list[j]
            st.markdown(_card_html(strat), unsafe_allow_html=True)
            if st.button("Lancer →", key=strat_id, use_container_width=True):
                st.session_state.selected_strategy = strat_id
                st.rerun()
    with c2:
        for j in [2, 3]:
            strat_id, strat = strats_list[j]
            st.markdown(_card_html(strat), unsafe_allow_html=True)
            if st.button("Lancer →", key=strat_id, use_container_width=True):
                st.session_state.selected_strategy = strat_id
                st.rerun()

    st.markdown("---")
    st.caption("Yahoo Finance • Walk-Forward • Monte Carlo • Bootstrap • Page Live pour signaux temps réel")
    st.stop()

# ============ PAGE STRATÉGIE (backtest détaillé) ============
strat_id = st.session_state.selected_strategy
if strat_id not in STRATEGIES:
    st.session_state.selected_strategy = None
    st.rerun()
strat = STRATEGIES[strat_id]

if st.button("← Retour"):
    st.session_state.selected_strategy = None
    st.rerun()

# ============ PAGE LIVE (stratégie lancée, PnL quotidien, config figée) ============
if strat_id == "live":
    import json
    import os

    LIVE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_config.json")
    LIVE_TICKERS = ["JPM", "XOM", "PG", "JNJ", "HON"]
    INITIAL_CAPITAL = 10000

    # Charger ou créer la config (date de lancement figée, tickers figés)
    if os.path.exists(LIVE_CONFIG_PATH):
        with open(LIVE_CONFIG_PATH, "r", encoding="utf-8") as f:
            live_config = json.load(f)
    else:
        live_config = {
            "launch_date": datetime.now().strftime("%Y-%m-%d"),
            "tickers": LIVE_TICKERS,
            "initial_capital": INITIAL_CAPITAL,
        }
        with open(LIVE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(live_config, f, indent=2)

    st.markdown(f"## {strat['icon']} {strat['name']}")
    st.caption(strat["description"])
    st.divider()
    st.info(f"**Stratégie lancée le {live_config['launch_date']}** · Tickers : {', '.join(live_config['tickers'])} · Capital initial : {INITIAL_CAPITAL:,.0f} € · *Configuration figée, mise à jour automatique chaque visite.*")

    since_live = datetime.strptime(live_config["launch_date"], "%Y-%m-%d")
    until_live = datetime.now()
    symbols_live = live_config["tickers"]

    with st.spinner("Chargement des données et calcul en cours..."):
        try:
            df_live, report_live = run_backtest_portfolio_hf(symbols=symbols_live, timeframe="1d", since=since_live, until=until_live, commission_pct=0.001, slippage_pct=0.0002)

            if df_live.empty or len(df_live) == 0:
                st.warning("Pas encore de données pour la période. Réessaie demain après la clôture des marchés.")
                st.stop()

            eq = df_live["strategy_equity_net"].values
            scale = INITIAL_CAPITAL / (eq[0] or 1)
            eq_eur = eq * scale
            current_val = float(eq_eur[-1])
            pnl_eur = current_val - INITIAL_CAPITAL
            pnl_pct = (current_val / INITIAL_CAPITAL - 1) * 100

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Valeur actuelle", f"{current_val:,.0f} €", "")
            with c2:
                st.metric("PnL", f"{pnl_eur:+,.0f} €", f"{pnl_pct:+.1f}%")
            with c3:
                st.metric("Rendement total", f"{pnl_pct:+.1f}%", "")
            with c4:
                st.metric("Dernière MAJ", df_live.index[-1].strftime("%Y-%m-%d"), "")

            df_plot = df_live.copy()
            df_plot["equity_eur"] = df_plot["strategy_equity_net"] * scale
            if "bh_equity" in df_plot.columns:
                df_plot["bh_eur"] = df_plot["bh_equity"] * (INITIAL_CAPITAL / df_plot["bh_equity"].iloc[0])
            if "sp500_equity" in df_plot.columns:
                df_plot["sp500_eur"] = df_plot["sp500_equity"] * (INITIAL_CAPITAL / df_plot["sp500_equity"].iloc[0])

            st.markdown("### Évolution du capital (€)")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df_plot.index, df_plot["equity_eur"], label="Stratégie", color="darkblue", linewidth=2)
            if "bh_eur" in df_plot.columns:
                ax.plot(df_plot.index, df_plot["bh_eur"], label="B&H", color="gray", alpha=0.8)
            if "sp500_eur" in df_plot.columns:
                ax.plot(df_plot.index, df_plot["sp500_eur"], label="S&P 500", color="#475569", linestyle="-.")
            ax.axhline(INITIAL_CAPITAL, color="red", linestyle="--", alpha=0.5, label="Capital initial")
            ax.set_ylabel("Valeur (€)")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("### Métriques")
            st.dataframe(pd.DataFrame({
                "Métrique": ["Rendement", "Sharpe", "Max DD", "Volatilité", "Win Rate"],
                "Valeur": [
                    f"{report_live.get('total_return_pct', 0):.1f}%",
                    f"{report_live.get('sharpe_ratio', 0):.2f}",
                    f"{report_live.get('max_drawdown_pct', 0):.1f}%",
                    f"{report_live.get('volatility_ann_pct', 0):.1f}%",
                    f"{report_live.get('period_win_rate_pct', report_live.get('win_rate_pct', 0)):.0f}%",
                ],
            }), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.exception(e)
    st.stop()

# Paramètres selon la stratégie
use_rsi_filter = strat_id == "sma_crossover_rsi"
is_buy_hold = strat_id == "buy_hold"
is_portfolio_hf = strat_id == "portfolio_hf"

# ============ SIDEBAR ============
with st.sidebar:
    st.title(f"{strat['icon']} {strat['name']}")

    symbol = st.text_input("Action (ticker)", value="AAPL", placeholder="ex: AAPL, TSLA")
    timeframe = st.selectbox("Timeframe", ["1d", "1wk", "1mo"], index=0)

    end_d = datetime.now()
    start_d = end_d - timedelta(days=365)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Début", value=start_d)
    with col2:
        end_date = st.date_input("Fin", value=end_d)

    st.subheader("Stratégie")
    sma_fast, sma_slow = 20, 50
    rsi_period, use_volume_filter, volume_ma_period = 14, False, 20
    portfolio_symbols = ["AAPL", "MSFT", "GOOGL"]
    portfolio_weights = [0.5, 0.3, 0.2]
    volatility_filter = True

    if is_buy_hold:
        st.info("Aucun paramètre : toujours en position long.")
    elif is_portfolio_hf:
        portfolio_tickers = st.text_input(
            "Actions (min 2, séparées par des virgules)",
            value="AAPL, MSFT, GOOGL, AMZN, NVDA",
            placeholder="ex: AAPL, MSFT, GOOGL, AMZN, NVDA",
            key="hf_symbols",
        )
        portfolio_symbols = [s.strip().upper() for s in portfolio_tickers.replace(";", ",").split(",") if s.strip()]
        st.caption("Période 1 : optimisation des facteurs (max SR) · Période 2 : validation")
        with st.expander("Paramètres fixes (lookbacks, Markowitz, risk)", expanded=True):
            factor_momentum_lb = st.number_input("Momentum lookback", value=20, min_value=5, max_value=100, key="hf_mom")
            factor_value_lb = st.number_input("Trend lookback (prix vs SMA)", value=50, min_value=10, max_value=200, key="hf_val")
            factor_vol_lb = st.number_input("Low vol lookback", value=20, min_value=5, max_value=100, key="hf_vol")
            markowitz_lookback = st.number_input("Lookback Markowitz", value=252, min_value=60, max_value=504, key="hf_mk")
            rebalance_bars = st.number_input("Rebalance (barres)", value=21, min_value=5, max_value=63, key="hf_rebal")
            var_limit_pct = st.number_input("Limite VaR % (0 = désactivé)", value=2.0, min_value=0.0, max_value=10.0, step=0.5, key="hf_var")
            max_correlation = st.slider("Max corrélation entre actifs", 0.5, 0.99, 0.85, key="hf_corr")
            max_weight_per_asset = st.slider("Max poids par actif", 0.1, 0.8, 0.4, key="hf_maxw")
            max_dd_circuit = st.number_input("Circuit breaker DD % (0 = désactivé)", value=12.0, min_value=0.0, max_value=50.0, step=1.0, key="hf_cb")
        with st.expander("Régime, Vol targeting, Turnover", expanded=False):
            use_regime_detection = st.checkbox("Détection régime marché", value=True, key="hf_regime")
            regime_lookback = st.number_input("Régime lookback", value=63, min_value=20, max_value=126, key="hf_reg_lb")
            regime_bear_scale = st.slider("Scale bear market", 0.2, 1.0, 0.5, key="hf_bear")
            regime_range_scale = st.slider("Scale range market", 0.5, 1.0, 0.75, key="hf_range")
            vol_target_ann_pct = st.number_input("Vol targeting % (0 = désactivé)", value=15.0, min_value=0.0, max_value=30.0, step=1.0, key="hf_vol_t")
            turnover_threshold = st.number_input("Turnover min % (0 = désactivé)", value=0.03, min_value=0.0, max_value=0.2, step=0.01, key="hf_to_thr")
            max_turnover_per_rebalance = st.number_input("Max turnover/rebalance (0 = désactivé)", value=0.3, min_value=0.0, max_value=1.0, step=0.1, key="hf_to_max")
            use_ir_objective = st.checkbox("Markowitz max IR (sinon Sharpe)", value=True, key="hf_ir")
            rolling_windows = st.number_input("Walk-forward glissant (fenêtres)", value=1, min_value=1, max_value=5, key="hf_roll")
    else:  # sma_crossover_rsi
        sma_fast = st.number_input("SMA rapide", value=20, min_value=2, max_value=100)
        sma_slow = st.number_input("SMA lente", value=50, min_value=2, max_value=200)
        stop_loss_pct = st.number_input("Stop Loss %", value=5.0, min_value=0.0, max_value=20.0, step=0.5, key="sma_sl")
        take_profit_pct = st.number_input("Take Profit %", value=15.0, min_value=0.0, max_value=50.0, step=0.5, key="sma_tp")

    short_allowed = False
    use_walk_forward = True
    commission_pct = 0.001
    slippage_pct = 0.0002
    in_sample_pct = 0.6
    rsi_long_max, rsi_short_min = 70.0, 30.0

    with st.expander("Paramètres avancés", expanded=False):
        if strat_id == "sma_crossover_rsi":
            short_allowed = st.checkbox("Short autorisé", value=False)
            rsi_period = st.number_input("Période RSI", value=14, key="rsi_sma")
            rsi_long_max = st.slider("RSI max achat", 50.0, 90.0, 70.0)
            rsi_short_min = st.slider("RSI min short", 10.0, 50.0, 30.0)
            use_volume_filter = st.checkbox("Filtre Volume", value=False, key="vol_sma")
            volume_ma_period = st.number_input("MA Volume", value=20, key="vol_ma") if use_volume_filter else 20
        commission_pct = st.slider("Commission %", 0.0, 0.5, 0.1) / 100
        slippage_pct = st.slider("Slippage %", 0.0, 0.1, 0.03) / 100
        spread_pct = st.slider("Spread % (bid-ask)", 0.0, 0.1, 0.02) / 100
        use_walk_forward = st.checkbox("Walk-Forward", value=True)
        in_sample_pct = st.slider("% In-Sample", 0.5, 0.8, 0.6)

    run_btn = st.button("▶ LANCER LE BACKTEST", type="primary", use_container_width=True)

# ============ MAIN (si pas encore lancé) ============
if not run_btn:
    st.info("👈 Configure les paramètres à gauche et clique sur **LANCER LE BACKTEST**")
    st.stop()

# Normalisation des tickers
if not is_portfolio_hf:
    symbol = (symbol or "AAPL").strip().upper().split(",")[0].strip() or "AAPL"

# ============ EXÉCUTION BACKTEST ============
since = datetime.combine(start_date, datetime.min.time())
until = datetime.combine(end_date, datetime.max.time())
periods_ppy = {"1d": 365, "1wk": 52, "1mo": 12}.get(timeframe, 365)

def _run_bt(wf_fn, bt_fn, wf_kwargs, bt_kwargs):
    if use_walk_forward and (until - since).days > 180:
        wf = wf_fn(**wf_kwargs)
        return wf["out_of_sample"]["df"], wf["out_of_sample"]["report"], wf["in_sample"]["report"], wf["in_sample"]["df"]
    df, report = bt_fn(**bt_kwargs)
    return df, report, None, None

with st.spinner("Backtest en cours..."):
    try:
        report_is = None
        df = None
        report = None
        df_in = None
        wf_opt_params = None
        base_kw = dict(symbol=symbol, timeframe=timeframe, since=since, until=until, commission_pct=commission_pct, slippage_pct=slippage_pct, spread_pct=spread_pct)
        bt_kw = dict(limit=1000, plot=False, verbose=False)

        if is_buy_hold:
            df, report, report_is, df_in = _run_bt(
                run_walk_forward_backtest_buy_hold, run_backtest_buy_hold,
                {**base_kw, "in_sample_pct": in_sample_pct},
                {**base_kw, **bt_kw},
            )
            df_full = run_backtest_buy_hold(**{**base_kw, **bt_kw})[0] if (use_walk_forward and (until - since).days > 180 and report_is is not None) else None
        elif is_portfolio_hf and len(portfolio_symbols) >= 2:
            hf_kw = dict(
                symbols=portfolio_symbols,
                timeframe=timeframe,
                factor_momentum_lb=factor_momentum_lb,
                factor_value_lb=factor_value_lb,
                factor_vol_lb=factor_vol_lb,
                markowitz_lookback=markowitz_lookback,
                rebalance_bars=rebalance_bars,
                var_limit_pct=var_limit_pct if var_limit_pct > 0 else None,
                max_correlation=max_correlation,
                max_weight_per_asset=max_weight_per_asset,
                max_dd_circuit_breaker=max_dd_circuit if max_dd_circuit > 0 else None,
                use_regime_detection=use_regime_detection,
                regime_lookback=regime_lookback,
                regime_bear_scale=regime_bear_scale,
                regime_range_scale=regime_range_scale,
                vol_target_ann_pct=vol_target_ann_pct if vol_target_ann_pct > 0 else None,
                turnover_threshold=turnover_threshold if turnover_threshold > 0 else None,
                max_turnover_per_rebalance=max_turnover_per_rebalance if max_turnover_per_rebalance > 0 else None,
                use_ir_objective=use_ir_objective,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
                spread_pct=spread_pct,
            )
            if use_walk_forward and (until - since).days > 180:
                wf = run_walk_forward_backtest_portfolio_hf(since=since, until=until, in_sample_pct=in_sample_pct, optimize_factors=True, rolling_windows=rolling_windows, optimize_objective="ir" if use_ir_objective else "sharpe", turnover_penalty=0.1, **hf_kw)
                df, report = wf["out_of_sample"]["df"], wf["out_of_sample"]["report"]
                report_is = wf["in_sample"]["report"]
                df_in = wf["in_sample"]["df"]
                wf_opt_params = wf.get("optimized_params")
                opt_kw = wf_opt_params if wf_opt_params else hf_kw
                df_full, _ = run_backtest_portfolio_hf(symbols=portfolio_symbols, timeframe=timeframe, since=since, until=until, **{k: v for k, v in opt_kw.items() if k not in ("symbols", "timeframe")})
            else:
                df, report = run_backtest_portfolio_hf(symbols=portfolio_symbols, timeframe=timeframe, since=since, until=until, **hf_kw)
                report_is = None
                df_in = None
                df_full = None
        elif is_portfolio_hf:
            st.error("Sélectionne au moins 2 actions pour le Backtesting Portfolio.")
            st.stop()
        else:  # sma_crossover_rsi
            kw = {**base_kw, "sma_fast": sma_fast, "sma_slow": sma_slow, "short_allowed": short_allowed,
                  "use_rsi_filter": use_rsi_filter, "rsi_period": rsi_period,
                  "rsi_long_max": rsi_long_max, "rsi_short_min": rsi_short_min,
                  "use_volume_filter": use_volume_filter, "volume_ma_period": volume_ma_period,
                  "stop_loss_pct": stop_loss_pct if stop_loss_pct > 0 else None,
                  "take_profit_pct": take_profit_pct if take_profit_pct > 0 else None}
            if use_walk_forward and (until - since).days > 180:
                wf = run_walk_forward_backtest(**{**kw, "in_sample_pct": in_sample_pct})
                df, report, report_is = wf["out_of_sample"]["df"], wf["out_of_sample"]["report"], wf["in_sample"]["report"]
                df_in = wf["in_sample"]["df"]
                df_full, _ = run_backtest(**{**kw, **bt_kw})
            else:
                df, report = run_backtest(**{**kw, **bt_kw})
                report_is = None
                df_in = None
                df_full = None

        pro = run_pro_analysis(df, periods_per_year=periods_ppy, n_monte_carlo=500, n_bootstrap=500)

    except Exception as e:
        st.error(f"Erreur : {e}")
        st.exception(e)
        st.stop()

# ============ DASHBOARD ============
st.markdown("---")
st.markdown("## Résumé")

# Ligne 1 : Contexte
ctx = f"{', '.join(portfolio_symbols)}" if is_portfolio_hf else symbol
st.markdown(f"**{ctx}** · {timeframe} · {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
if is_portfolio_hf:
    st.caption("B&H = portefeuille équipondéré des actions · S&P 500 = benchmark marché")

# Ligne 2 : Walk-Forward (masqué pour B&H)
if not is_buy_hold and use_walk_forward and (until - since).days > 180 and report_is is not None:
    st.markdown("**Période 1** (optimisation) → **Période 2** (validation)")
    if is_portfolio_hf and wf_opt_params:
        st.caption(f"Facteurs optimisés : Mom {wf_opt_params.get('factor_w_momentum', 0):.2f} · Trend {wf_opt_params.get('factor_w_value', 0):.2f} · LowVol {wf_opt_params.get('factor_w_vol', 0):.2f} · Seuil {wf_opt_params.get('factor_threshold', 0):.2f}")

# Rapport utilisé : toute la période si dispo (cohérence avec "Toute période")
r_main = compute_risk_report(df_full, periods_per_year=periods_ppy) if df_full is not None else report

ret_s, ret_bh = r_main["total_return_pct"], r_main["bh_return_pct"]
sharpe_s, sharpe_bh = r_main["sharpe_ratio"], r_main.get("sharpe_ratio_bh", 0) or 0
dd_s = r_main["max_drawdown_pct"]
dd_bh = r_main.get("bh_max_drawdown_pct", 0) or 0
vol_s = r_main.get("volatility_ann_pct", 0) or 0
vol_bh = r_main.get("bh_volatility_ann_pct", 0) or 0
ir = r_main.get("information_ratio", 0) or 0
wr_s = r_main.get("period_win_rate_pct", r_main.get("win_rate_pct", 0)) or 0
wr_bh = r_main.get("bh_win_rate_pct", 0) or 0

# Verdict et comparaison : masqués pour Buy & Hold (pas de référence à comparer)
if not is_buy_hold:
    better_sharpe = sharpe_s > sharpe_bh
    better_dd = dd_s > dd_bh
    better_vol = vol_s < vol_bh
    better_ir = ir > 0
    score = sum([better_sharpe, better_dd, better_vol, better_ir])
    violated = []
    if not better_sharpe: violated.append("Sharpe")
    if not better_dd: violated.append("Max DD")
    if not better_vol: violated.append("Volatilité")
    if not better_ir: violated.append("IR")
    violated_str = ", ".join(violated) if violated else "aucun"
    apply_ok = score >= 3 and ret_s > 0 and (report_is is None or (report_is["total_return_pct"] > 0 and report["total_return_pct"] > 0))
    if apply_ok:
        msg = f"**✅ La stratégie est meilleure que B&H** ({score}/4 critères) · **Recommandation : Appliquer**"
        if violated:
            msg += f" · Critères violés : {violated_str}"
        st.success(msg)
    else:
        st.warning(f"**⚠️ La stratégie n'est pas meilleure que B&H** ({score}/4 critères) · Critères violés : {violated_str} · **Recommandation : Ne pas appliquer**")

# Métriques : tableau simple pour B&H (sans comparaison), comparaison Strat vs B&H vs S&P pour les autres
st.markdown("### Stratégie vs B&H vs S&P 500" if not is_buy_hold else "### Métriques")
if df_full is not None and not is_buy_hold:
    st.caption("Métriques sur toute la période (cohérent avec « Toute période »)")
if is_buy_hold:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Rendement", f"{ret_s:.1f}%", None)
    with col2:
        st.metric("Sharpe", f"{sharpe_s:.2f}", None)
    with col3:
        st.metric("Max Drawdown", f"{dd_s:.1f}%", None)
    with col4:
        st.metric("Volatilité ann.", f"{vol_s:.1f}%", None)
    with col5:
        st.metric("Win Rate", f"{wr_s:.0f}%", None)
elif (has_sp500 := is_portfolio_hf and "sp500_return_pct" in r_main):
    ret_sp = r_main.get("sp500_return_pct", 0) or 0
    sharpe_sp = r_main.get("sp500_sharpe", 0) or 0
    dd_sp = r_main.get("sp500_max_drawdown_pct", 0) or 0
    vol_sp = r_main.get("sp500_volatility_ann_pct", 0) or 0
    wr_sp = r_main.get("sp500_win_rate_pct", 0) or 0
    ir_sp = r_main.get("information_ratio_vs_sp500", 0) or 0
    tbl = pd.DataFrame({
        "Métrique": ["Rendement", "Sharpe", "Max DD", "Volatilité", "Win Rate", "IR (vs B&H)", "IR (vs S&P 500)"],
        "Strat": [f"{ret_s:.1f}%", f"{sharpe_s:.2f}", f"{dd_s:.1f}%", f"{vol_s:.1f}%", f"{wr_s:.0f}%", f"{ir:.2f}", f"{ir_sp:.2f}"],
        "B&H (portefeuille)": [f"{ret_bh:.1f}%", f"{sharpe_bh:.2f}", f"{dd_bh:.1f}%", f"{vol_bh:.1f}%", f"{wr_bh:.0f}%", "—", "—"],
        "S&P 500 (benchmark)": [f"{ret_sp:.1f}%", f"{sharpe_sp:.2f}", f"{dd_sp:.1f}%", f"{vol_sp:.1f}%", f"{wr_sp:.0f}%", "—", "—"],
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)
else:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Rendement", f"Strat {ret_s:.1f}%", f"B&H {ret_bh:.1f}%")
    with col2:
        st.metric("Sharpe", f"Strat {sharpe_s:.2f}", f"B&H {sharpe_bh:.2f}")
    with col3:
        st.metric("Max Drawdown", f"Strat {dd_s:.1f}%", f"B&H {dd_bh:.1f}%")
    with col4:
        st.metric("Volatilité ann.", f"Strat {vol_s:.1f}%", f"B&H {vol_bh:.1f}%")
    with col5:
        st.metric("Win Rate", f"Strat {wr_s:.0f}%", f"B&H {wr_bh:.0f}%")
    with col6:
        st.metric("IR (vs B&H)", f"{ir:.2f}", "> 0 = mieux")
if not is_buy_hold:
    st.caption(f"Verdict Walk-Forward : {'Robuste' if (report_is and report_is['total_return_pct'] > 0 and report['total_return_pct'] > 0) else ('Surajustement' if report_is else '—')}")

# Walk-Forward : 3 périodes (masqué pour B&H)
if not is_buy_hold and use_walk_forward and (until - since).days > 180 and report_is is not None:
    st.markdown("### Par période")
    c_is, c_oos, c_all = st.columns(3)
    with c_is:
        st.metric("Période 1 (optim.)", f"{report_is['total_return_pct']:.1f}%", f"Sharpe {report_is['sharpe_ratio']:.2f}")
    with c_oos:
        st.metric("Période 2 (valid.)", f"{report['total_return_pct']:.1f}%", f"Sharpe {report['sharpe_ratio']:.2f}")
    with c_all:
        if df_full is not None:
            st.metric("Toute période", f"{r_main['total_return_pct']:.1f}%", f"Sharpe {r_main['sharpe_ratio']:.2f}")
        else:
            st.metric("Toute période", "—", "")

st.markdown("---")
st.markdown("## Graphiques")

def _main_chart(d, title, suffix=""):
    """Un seul graphique : Performance Strat vs B&H + Drawdown (ou courbe seule pour B&H)."""
    close_cols = [c for c in d.columns if c.startswith("Close_")] if is_portfolio_hf else None
    return get_plot_figure(d, title=title, symbol=symbol if not is_portfolio_hf else "Portfolio", close_cols=close_cols, dark_theme=False, compact=is_portfolio_hf, force_show_brute=is_portfolio_hf, hide_comparison=is_buy_hold)

if is_buy_hold:
    # B&H : un seul graphique (vue d'ensemble)
    d_show = df_full if df_full is not None else df
    st.pyplot(_main_chart(d_show, f"Performance · {d_show.index[0].strftime('%Y-%m-%d')} → {d_show.index[-1].strftime('%Y-%m-%d')}"))
elif df_in is not None:
    st.markdown("**Période 1** (optimisation)")
    st.pyplot(_main_chart(df_in, f"In-Sample · {df_in.index[0].strftime('%Y-%m-%d')} → {df_in.index[-1].strftime('%Y-%m-%d')}"))
    st.markdown("**Période 2** (validation)")
    st.pyplot(_main_chart(df, f"Out-of-Sample · {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"))
    if df_full is not None:
        st.markdown("**Vue d'ensemble** (toute la période)")
        st.pyplot(_main_chart(df_full, f"Toute période · {df_full.index[0].strftime('%Y-%m-%d')} → {df_full.index[-1].strftime('%Y-%m-%d')}"))
else:
    st.pyplot(_main_chart(df, f"Performance · {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"))

# Monte Carlo, Tests stat, Données : masqués pour B&H
if not is_buy_hold:
    tab1, tab2, tab3 = st.tabs(["Monte Carlo & Bootstrap", "Tests statistiques & Stress", "Données"])
    with tab1:
        if "error" not in pro:
            mc, bs = pro["monte_carlo"], pro["bootstrap"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Monte Carlo** (500 sim.)")
                st.metric("Proba rendement > 0", f"{mc['prob_positive_return']:.0f}%")
                st.caption(f"Rendement IC 90% : [{mc['final_return_5pct']:.1f}% ; {mc['final_return_95pct']:.1f}%]")
            with c2:
                st.markdown("**Bootstrap** (500 iter.)")
                st.metric("Rendement IC 95%", f"[{bs['return_ci_95_low']:.1f}% ; {bs['return_ci_95_high']:.1f}%]")
                st.caption(f"Sharpe IC 95% : [{bs['sharpe_ci_95_low']:.2f} ; {bs['sharpe_ci_95_high']:.2f}]")
            if len(pro["rolling_metrics"]) > 0:
                st.line_chart(pro["rolling_metrics"].rename(columns={
                    "rolling_sharpe": "Sharpe", "rolling_vol": "Vol %", "rolling_dd": "DD %"
                }))
    with tab2:
        if "error" not in pro:
            st_ = pro["statistical_tests"]
            stress = pro["stress_test"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Distribution**")
                st.write(f"Jarque-Bera p-value: {st_['jarque_bera_pvalue']:.4f} → {st_['interpretation']}")
                st.write(f"Skewness: {st_['skewness']:.2f} | Kurtosis: {st_['kurtosis']:.2f}")
            with c2:
                st.markdown("**Stress Test**")
                st.metric("Pire période", f"{stress['worst_single_return_pct']:.1f}%")
    with tab3:
        if is_portfolio_hf:
            close_cols = [c for c in df.columns if c.startswith("Close_")]
            base_cols = ["strategy_equity_net", "bh_equity"]
            if "sp500_equity" in df.columns:
                base_cols.append("sp500_equity")
            disp = df[base_cols + close_cols].copy()
            disp.columns = (["Equity Strat.", "Equity B&H"] + (["S&P 500"] if "sp500_equity" in df.columns else []) + [c.replace("Close_", "") for c in close_cols])
        elif "SMA_fast" in df.columns:
            disp = df[["Close", "SMA_fast", "SMA_slow", "signal", "strategy_equity_net", "bh_equity"]].copy()
            disp.columns = ["Close", "SMA Fast", "SMA Slow", "Signal", "Equity Strat.", "Equity B&H"]
        else:
            disp = df[["Close", "signal", "strategy_equity_net", "bh_equity"]].copy()
            disp.columns = ["Close", "Signal", "Equity Strat.", "Equity B&H"]
        st.dataframe(disp, use_container_width=True)
        fn = f"backtest_portfolio_{'_'.join(s.replace('/', '') for s in portfolio_symbols)}.csv" if is_portfolio_hf else f"backtest_{symbol.replace('/', '_')}.csv"
        st.download_button("📥 CSV", disp.to_csv(), file_name=fn, mime="text/csv")
