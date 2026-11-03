"""Turns raw prices into a scored panel: one row per asset per day."""
import numpy as np
import pandas as pd

from config import HORIZONS, MARKETS
from model import (add_rs_rank, features, hist_winrate, market_trade, regime_series, scores,
                   simulate)
from universe import market_of


def build(prices, meta, mkt_prices):
    """prices: {sym: OHLCV}; meta: {sym: info}; mkt_prices: {market: OHLCV of its index}.
    Returns (feats {sym: DataFrame}, panel DataFrame, mom DataFrame)."""
    feats = {s: features(df) for s, df in prices.items()}
    mom = add_rs_rank(feats)
    regimes = {m: regime_series(df) for m, df in mkt_prices.items()}
    rows = []
    for s, f in feats.items():
        mk = meta.get(s, {}).get("market") or market_of(s)
        reg = regimes.get(mk, regimes.get("US"))
        scores(f, reg)
        a = f["atr"].to_numpy()
        df = prices[s]
        for h in HORIZONS:
            for d, tag in ((1, "l"), (-1, "s")):
                ret, out, xi = simulate(df, a, d, h)
                f[f"ret_{tag}{h}"] = ret
                f[f"out_{tag}{h}"] = out
                f[f"xi_{tag}{h}"] = xi
                xd = np.full(len(xi), np.datetime64("NaT"), dtype="datetime64[ns]")
                okx = xi >= 0
                xd[okx] = f.index.to_numpy()[xi[okx]]
                f[f"xd_{tag}{h}"] = xd
                sig = (f["dir"].to_numpy() == d)
                w, c, shr = hist_winrate(sig, ret, xi)
                f[f"wn_{tag}{h}"] = w
                f[f"n_{tag}{h}"] = c
                f[f"wr_{tag}{h}"] = shr
            mdf = mkt_prices.get(mk)
            if mdf is None:
                mdf = mkt_prices.get("US")
            f[f"mkt{h}"] = market_trade(mdf, f.index, h) if mdf is not None else np.nan
        p = f.drop(columns=[c for c in f.columns if c.startswith("xi_")]).copy()
        p["sym"] = s
        p["market"] = mk
        rows.append(p.reset_index(names="date"))
    panel = pd.concat(rows, ignore_index=True)
    num = panel.select_dtypes("float64").columns
    panel[num] = panel[num].astype("float32")
    panel = panel[panel["score_long"].notna()]
    return feats, panel, mom


def market_frames(download_fn, extra=()):
    syms = list(MARKETS.values())
    got = download_fn(syms + list(extra))
    mk = {m: got[s] for m, s in MARKETS.items() if s in got}
    if "IL" not in mk and "US" in mk:          # TA-125 index sometimes missing on Yahoo
        mk["IL"] = mk["US"]
    return mk, got
