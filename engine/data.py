"""Price data: Yahoo Finance via yfinance (free). SYNTHETIC=1 → random-walk test data."""
import os

import numpy as np
import pandas as pd

from config import HISTORY_YEARS, MIN_BARS

SYNTHETIC = os.environ.get("SYNTHETIC") == "1"


def _clean(df):
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df["Close"] > 0) & (df["Open"] > 0)]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["Volume"] = df["Volume"].fillna(0)
    return df


def download(symbols, years=HISTORY_YEARS, batch=100):
    """Return {symbol: OHLCV DataFrame} for symbols with enough history."""
    if SYNTHETIC:
        return synthetic(symbols, years)
    import yfinance as yf

    out = {}
    symbols = list(dict.fromkeys(symbols))
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            raw = yf.download(chunk, period=f"{years}y", interval="1d", auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
        except Exception as e:
            print("download failed for batch", i, e)
            continue
        for s in chunk:
            try:
                df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = _clean(df)
                # crypto trades 24/7: today's daily candle is still forming — drop it
                if s.endswith("-USD") and len(df) and df.index[-1] >= pd.Timestamp.utcnow().tz_localize(None).normalize():
                    df = df.iloc[:-1]
                if len(df) >= MIN_BARS:
                    out[s] = df
            except Exception:
                pass
    print(f"downloaded {len(out)}/{len(symbols)} symbols")
    return out


def synthetic(symbols, years):
    """Deterministic fake markets for offline testing only — never used in production."""
    out = {}
    n = 252 * years
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    mkt = np.random.default_rng(7).normal(0.0003, 0.01, n)
    for s in symbols:
        rng = np.random.default_rng(abs(hash(s)) % (2**32))
        drift = rng.normal(0.0003, 0.0004)
        beta = rng.uniform(0.6, 1.4)
        vol = rng.uniform(0.012, 0.03)
        # regime-y momentum: persistent idiosyncratic trend
        trend = np.cumsum(rng.normal(0, 0.00015, n))
        r = drift + beta * mkt + trend * 0.3 + rng.normal(0, vol, n)
        close = 50 * np.exp(np.cumsum(r))
        openp = close * np.exp(rng.normal(0, vol / 3, n))
        high = np.maximum(openp, close) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
        low = np.minimum(openp, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
        volume = rng.lognormal(14, 0.4, n) * (1 + np.abs(r) * 20)
        out[s] = pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close,
                               "Volume": volume}, index=dates)
    return out
