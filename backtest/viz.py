"""
Visualizer : graphiques de performance et drawdown.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def get_price_chart_figure(
    df: pd.DataFrame,
    title: str = "Prix + Signaux",
    show_signals: bool = True,
    strategy_type: str = "sma",
    dark_theme: bool = False,
):
    """Graphique : Prix + indicateurs (SMA ou Bollinger) + marqueurs d'entrée/sortie."""
    fig, ax = plt.subplots(figsize=(14, 6))
    price_color = "#e2e8f0" if dark_theme else "black"
    ax.plot(df.index, df["Close"], label="Price", color=price_color, alpha=0.85, linewidth=1)

    if strategy_type == "price":
        pass
    elif strategy_type == "bollinger" and "BB_upper" in df.columns:
        ax.plot(df.index, df["BB_upper"], label="Bande haute", color="red", alpha=0.6, linewidth=1)
        ax.plot(df.index, df["BB_middle"], label="Bande milieu", color="gray", alpha=0.6, linewidth=1)
        ax.plot(df.index, df["BB_lower"], label="Bande basse", color="green", alpha=0.6, linewidth=1)
    elif strategy_type == "macd" and "MACD" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df.index, df["MACD"], label="MACD", color="blue", alpha=0.8, linewidth=1.5)
        ax2.plot(df.index, df["MACD_signal"], label="Signal", color="orange", alpha=0.8, linewidth=1)
        ax2.set_ylabel("MACD")
        ax2.legend(loc="upper right", fontsize=8)
    elif strategy_type == "rsi" and "RSI" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df.index, df["RSI"], label="RSI", color="purple", alpha=0.8, linewidth=1)
        ax2.axhline(y=30, color="green", linestyle="--", alpha=0.5)
        ax2.axhline(y=70, color="red", linestyle="--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right", fontsize=8)
    elif "SMA_fast" in df.columns:
        ax.plot(df.index, df["SMA_fast"], label="Fast SMA", color="green", alpha=0.8, linewidth=1.5)
        ax.plot(df.index, df["SMA_slow"], label="Slow SMA", color="orange", alpha=0.8, linewidth=1.5)

    if show_signals and "signal" in df.columns:
        entries = df[df["signal"].diff() == 1]
        exits = df[df["signal"].diff() == -1]
        if len(entries) > 0:
            ax.scatter(entries.index, entries["Close"], color="green", marker="^", s=80, zorder=5, label="Long entry")
        if len(exits) > 0:
            ax.scatter(exits.index, exits["Close"], color="red", marker="v", s=80, zorder=5, label="Exit")

    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    if dark_theme:
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#e2e8f0")
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                t.set_color("#e2e8f0")
    plt.tight_layout()
    return fig


def get_plot_figure(
    df: pd.DataFrame,
    title: str = "Backtest: Performance & Drawdown",
    symbol: str = "",
    close_cols: Optional[list] = None,
    dark_theme: bool = False,
    compact: bool = False,
    force_show_brute: bool = False,
    hide_comparison: bool = False,
):
    """Returns a matplotlib figure. compact=True: Performance + Drawdown only (no price)."""
    if compact or (close_cols and len(close_cols) > 3):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True)
        axes = [axes[0], axes[1]]
        skip_price = True
    else:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[1, 2, 1], sharex=True)
        skip_price = False
    if dark_theme:
        fig.patch.set_facecolor("#0f172a")
        line_colors = ["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#22d3ee", "#fb923c", "#4ade80"]
    else:
        line_colors = list(plt.cm.tab10.colors)

    if not skip_price:
        ax0 = axes[0]
        if close_cols and len(close_cols) > 0:
            colors = line_colors if dark_theme else list(plt.cm.tab10.colors)
            for i, col in enumerate(close_cols):
                lbl = col.replace("Close_", "").replace("_", "/")
                ax0.plot(df.index, df[col], label=lbl, color=colors[i % len(colors)], alpha=0.8, linewidth=1.5)
                sig_col = f"signal_{col.replace('Close_', '')}"
                if sig_col in df.columns:
                    sig = df[sig_col].fillna(0).astype(int)
                    entries = df[sig.diff() == 1]
                    exits = df[sig.diff() == -1]
                    max_markers = 120
                    if len(entries) > max_markers:
                        entries = entries.iloc[:: max(1, len(entries) // max_markers)]
                    if len(exits) > max_markers:
                        exits = exits.iloc[:: max(1, len(exits) // max_markers)]
                    if len(entries) > 0:
                        ax0.scatter(entries.index, entries[col], color="green", marker="^", s=50, zorder=5, alpha=0.85)
                    if len(exits) > 0:
                        ax0.scatter(exits.index, exits[col], color="red", marker="v", s=50, zorder=5, alpha=0.85)
        else:
            c0 = "#e2e8f0" if dark_theme else "black"
            ax0.plot(df.index, df["Close"], label=f"Price {symbol or 'asset'}", color=c0, alpha=0.85, linewidth=1.5)
            if "signal" in df.columns:
                entries = df[df["signal"].diff() == 1]
                exits = df[df["signal"].diff() == -1]
                if len(entries) > 0:
                    ax0.scatter(entries.index, entries["Close"], color="green", marker="^", s=80, zorder=5, label="Buy")
                if len(exits) > 0:
                    ax0.scatter(exits.index, exits["Close"], color="red", marker="v", s=80, zorder=5, label="Sell")
        ax0.set_ylabel("Price ($)")
        ax0.set_title("Price evolution")
        ax0.legend(loc="upper left", fontsize=9)
        ax0.grid(True, alpha=0.3)
        ax0.set_ylim(bottom=0)
        ax1 = axes[1]
    else:
        ax1 = axes[0]

    strat_c = "#60a5fa" if dark_theme else "darkblue"
    if not hide_comparison:
        bh_c = "#94a3b8" if dark_theme else "gray"
        sp500_c = "#64748b" if dark_theme else "#475569"
        ax1.plot(df.index, df["bh_equity"], label="B&H (portfolio)", color=bh_c, alpha=0.85, linewidth=1.5)
        if "sp500_equity" in df.columns:
            ax1.plot(df.index, df["sp500_equity"], label="S&P 500 (benchmark)", color=sp500_c, alpha=0.85, linewidth=1.5, linestyle="-.")
        show_brute = force_show_brute or (
            "strategy_equity" in df.columns
            and not np.allclose(df["strategy_equity"].values, df["strategy_equity_net"].values, rtol=1e-5)
        )
        if show_brute and "strategy_equity" in df.columns:
            ax1.plot(df.index, df["strategy_equity"], label="Gross strat. (no fees)", color="darkorange", alpha=0.95, linewidth=2, linestyle="--", zorder=3)
    ax1.plot(df.index, df["strategy_equity_net"], label="Strategy" if hide_comparison else "Strategy (with fees)", color=strat_c, linewidth=2)
    if hide_comparison and len(df) >= 2:
        ax1.scatter(df.index[0], df["strategy_equity_net"].iloc[0], color="green", marker="^", s=60, zorder=5, alpha=0.9, label="Buy")
        ax1.scatter(df.index[-1], df["strategy_equity_net"].iloc[-1], color="red", marker="v", s=60, zorder=5, alpha=0.9, label="Sell")
    elif "signal" in df.columns and not hide_comparison:
        sig = df["signal"].fillna(0).astype(int)
        entries = df[sig.diff() == 1]
        exits = df[sig.diff() == -1]
        max_m = 150
        for evt, label, color, m in [(entries, "Buy", "green", "^"), (exits, "Sell", "red", "v")]:
            if len(evt) > 0:
                sub = evt.iloc[:: max(1, len(evt) // max_m)] if len(evt) > max_m else evt
                ax1.scatter(sub.index, sub["strategy_equity_net"], color=color, marker=m, s=45, zorder=5, alpha=0.9, label=label)
    elif "turnover" in df.columns and not hide_comparison:
        rebal = df[df["turnover"] > 0.001]
        if len(rebal) > 0:
            sub = rebal.iloc[:: max(1, len(rebal) // 120)] if len(rebal) > 120 else rebal
            ax1.scatter(sub.index, sub["strategy_equity_net"], color="purple", marker="o", s=35, zorder=5, alpha=0.8, label="Rebalance")
    ax1.set_ylabel("Performance (× capital)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = axes[2] if not skip_price else axes[1]
    cummax = df["strategy_equity_net"].cummax()
    drawdown = (df["strategy_equity_net"] - cummax) / cummax
    ax2.fill_between(df.index, drawdown, 0, color="#ef4444" if dark_theme else "crimson", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(top=0)
    if dark_theme:
        all_axes = [ax1, ax2] if skip_price else axes
        for ax in all_axes:
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#475569")
            ax.xaxis.label.set_color("#e2e8f0")
            ax.yaxis.label.set_color("#e2e8f0")
            ax.title.set_color("#e2e8f0")
            if ax.get_legend():
                for t in ax.get_legend().get_texts():
                    t.set_color("#e2e8f0")
    plt.tight_layout()
    return fig


def plot_results(df: pd.DataFrame, output_path: str = "backtest_results.png"):
    """Saves the chart to disk."""
    fig = get_plot_figure(df)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {output_path}")
