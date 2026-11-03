"""Confluence Terminal — global settings. Edit here, not in the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"          # committed: picks history, backtest, extra tickers
SITE = ROOT / "site"            # deployed to GitHub Pages
SITE_DATA = SITE / "data"

HISTORY_YEARS = 6               # price history to download
MIN_BARS = 260                  # skip assets with less history

# Exit rules (ATR multiples). Target = 2R.
STOP_ATR = 1.5
TARGET_ATR = 3.0
HORIZONS = [1, 5, 10]           # max holding days tested in the backtest
DEFAULT_HORIZON = 10            # used if the backtest can't decide

COST = 0.0015                   # round-trip cost + slippage (0.15%)
SIGNAL_MIN = 60                 # score needed to call a direction
PICKS_PER_DAY = 5
RANDOM_SEEDS = 30               # random-entry benchmark repetitions
OOS_FRACTION = 0.3              # last 30% of dates = out-of-sample

CHART_BARS = 260                # bars stored per ticker for the chart

# Weights of the score components (sum = 1)
WEIGHTS = {
    "trend": 0.25,
    "rs": 0.25,
    "breakout": 0.20,
    "rsi": 0.10,
    "volume": 0.10,
    "regime": 0.10,
}

MARKETS = {"US": "SPY", "IL": "^TA125.TA", "CRYPTO": "BTC-USD"}
