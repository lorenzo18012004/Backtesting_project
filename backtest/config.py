"""
Constantes du moteur de backtesting.
"""

# Mapping timeframe → durée en secondes
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

# Periods per year by timeframe (for annualization)
PERIODS_PER_YEAR = {
    "1m": 525600, "5m": 105120, "15m": 35040,
    "1h": 8760, "4h": 2190, "1d": 365, "1wk": 52, "1mo": 12,
}
