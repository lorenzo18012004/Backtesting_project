"""
PnL Engine : calcul des rendements stratégie, frais et slippage.
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_strategy_returns(
    df: pd.DataFrame,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Rendement stratégie = signal * rendement actif
    Si stop_loss_pct / take_profit_pct fournis : applique SL/TP (sortie anticipée).
    """
    df = df.copy()

    use_sl = stop_loss_pct is not None and stop_loss_pct > 0
    use_tp = take_profit_pct is not None and take_profit_pct > 0
    if use_sl or use_tp:
        sl = stop_loss_pct if use_sl else 1.0
        tp = take_profit_pct if use_tp else 10.0
        strategy_returns = []
        entry_price = None
        position = 0

        for i in range(len(df)):
            sig = int(df["signal"].iloc[i])
            row = df.iloc[i]
            ret = 0.0

            if sig != 0 and position == 0:
                position = sig
                entry_price = row["Open"]

            if position != 0 and entry_price is not None:
                low, high = row["Low"], row["High"]
                if position == 1:
                    if use_sl and low <= entry_price * (1 - sl / 100):
                        ret = np.log((1 - sl / 100))
                        position = 0
                    elif use_tp and high >= entry_price * (1 + tp / 100):
                        ret = np.log(1 + tp / 100)
                        position = 0
                    else:
                        ret = row["log_return"]
                        if sig == 0:
                            position = 0
                elif position == -1:
                    if use_sl and high >= entry_price * (1 + sl / 100):
                        ret = np.log(1 - sl / 100)
                        position = 0
                    elif use_tp and low <= entry_price * (1 - tp / 100):
                        ret = np.log(1 + tp / 100)
                        position = 0
                    else:
                        ret = -row["log_return"]
                        if sig == 0:
                            position = 0

            strategy_returns.append(ret)

        df["strategy_return"] = strategy_returns
    else:
        df["strategy_return"] = df["signal"] * df["log_return"]
        df["strategy_return"] = df["strategy_return"].fillna(0)

    df["cum_log_return_strategy"] = df["strategy_return"].cumsum()
    df["strategy_equity"] = np.exp(df["cum_log_return_strategy"])

    df["cum_log_return_bh"] = df["log_return"].cumsum()
    df["bh_equity"] = np.exp(df["cum_log_return_bh"])

    return df


def apply_costs(
    df: pd.DataFrame,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0002,
    spread_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Détecte chaque changement de position (trade).
    Pénalité = commission + slippage + spread/2 par trade.
    """
    if df.empty or len(df) == 0:
        return df
    df = df.copy()
    df["position_change"] = df["signal"].diff().abs()

    cost_per_trade = commission_pct + slippage_pct + spread_pct / 2

    if (df["signal"] == 1).all():
        df["trade_cost"] = 0.0
        df.loc[df.index[0], "trade_cost"] = cost_per_trade
        df.loc[df.index[-1], "trade_cost"] = cost_per_trade
        df["position_change"] = 0.0
        df.loc[df.index[0], "position_change"] = 1.0
        df.loc[df.index[-1], "position_change"] = 1.0
    else:
        trades = df["position_change"] > 0
        df["trade_cost"] = np.where(trades, cost_per_trade, 0)
        if df["signal"].iloc[-1] != 0:
            df.loc[df.index[-1], "trade_cost"] = df.loc[df.index[-1], "trade_cost"] + cost_per_trade
            df.loc[df.index[-1], "position_change"] = df.loc[df.index[-1], "position_change"] + 1.0
    df["strategy_return_net"] = df["strategy_return"] - df["trade_cost"]

    df["cum_log_return_strategy_net"] = df["strategy_return_net"].cumsum()
    df["strategy_equity_net"] = np.exp(df["cum_log_return_strategy_net"])

    return df
