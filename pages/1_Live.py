"""
Dashboard Live - Signaux en temps réel
Les utilisateurs peuvent suivre les signaux de la stratégie en direct.
"""

import streamlit as st
from datetime import datetime
from backtest import get_live_signals

st.set_page_config(page_title="Live | Backtesting Pro", page_icon="🔴", layout="wide")
st.markdown("# 🔴 Live - Signaux en temps réel")
st.caption("Suivi des signaux SMA + RSI sur les actions sélectionnées. Données Yahoo Finance, mises à jour à chaque rafraîchissement.")

st.divider()

with st.sidebar:
    st.title("🔴 Live")
    symbols = st.multiselect(
        "Actions à suivre",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ", "WMT"],
        default=["AAPL", "MSFT", "GOOGL"],
    )
    timeframe = st.selectbox("Timeframe", ["1d", "1wk"], index=0)
    sma_fast = st.number_input("SMA rapide", value=20, min_value=2, max_value=100)
    sma_slow = st.number_input("SMA lente", value=50, min_value=2, max_value=200)
    use_rsi = st.checkbox("Filtre RSI", value=True)
    rsi_period = st.number_input("Période RSI", value=14) if use_rsi else 14
    rsi_long_max = st.slider("RSI max achat", 50.0, 90.0, 70.0) if use_rsi else 70.0
    rsi_short_min = st.slider("RSI min short", 10.0, 50.0, 30.0) if use_rsi else 30.0

if not symbols:
    st.warning("Sélectionne au moins une action.")
    st.stop()

if st.button("🔄 Rafraîchir les signaux"):
    st.rerun()

with st.spinner("Récupération des données..."):
    try:
        signals = get_live_signals(
            symbols=symbols,
            timeframe=timeframe,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            use_rsi_filter=use_rsi,
            rsi_period=rsi_period,
            rsi_long_max=rsi_long_max,
            rsi_short_min=rsi_short_min,
        )
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.stop()

st.success(f"✅ Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")

# Cartes par paire
cols = st.columns(min(len(symbols), 4))
for i, sym in enumerate(symbols):
    with cols[i % len(cols)]:
        data = signals.get(sym, {})
        if "error" in data:
            st.error(f"{sym}: {data['error']}")
            continue

        signal = data["signal"]
        price = data["price"]
        rsi = data.get("rsi")
        sig_label = "🟢 LONG" if signal == 1 else "🔴 CASH"
        sig_color = "green" if signal == 1 else "gray"

        with st.container(border=True):
            st.markdown(f"### {sym}")
            st.metric("Prix", f"${price:,.2f}")
            st.metric("Signal", sig_label)
            if rsi is not None:
                st.metric("RSI", f"{rsi:.1f}")
            st.caption(f"Dernière bougie : {data['last_update'].strftime('%Y-%m-%d %H:%M')}")

st.divider()
st.info("💡 **Astuce** : Clique sur « Rafraîchir les signaux » pour mettre à jour. Pour un suivi automatique, utilise un outil de rafraîchissement (ex. extension navigateur) ou déploie l'app avec Streamlit Cloud.")
