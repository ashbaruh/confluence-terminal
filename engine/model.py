"""Indicators, scores, trade simulation, historical win rate, reasoning text."""
import numpy as np
import pandas as pd

from config import COST, HORIZONS, SIGNAL_MIN, STOP_ATR, TARGET_ATR, WEIGHTS

SHRINK_K = 8  # pulls small-sample win rates toward 50%


# ---------- indicators ----------
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def features(df):
    f = pd.DataFrame(index=df.index)
    c = df["Close"]
    f["close"] = c
    f["sma20"] = c.rolling(20).mean()
    f["sma50"] = c.rolling(50).mean()
    f["sma200"] = c.rolling(200).mean()
    f["rsi"] = rsi(c)
    f["atr"] = atr(df)
    f["atrp"] = f["atr"] / c
    f["mom"] = 0.5 * c.pct_change(20) + 0.5 * c.pct_change(60)
    f["hi20"] = df["High"].rolling(20).max().shift(1)
    f["lo20"] = df["Low"].rolling(20).min().shift(1)
    vavg = df["Volume"].rolling(20).mean().shift(1)
    f["volr"] = (df["Volume"] / vavg.replace(0, np.nan)).fillna(1.0)
    f["slope50"] = f["sma50"] / f["sma50"].shift(10) - 1
    # research candidates
    f["rsi2"] = rsi(c, 2)
    f["ret5"] = c.pct_change(5)
    f["ret1"] = c.pct_change(1)
    f["dist20"] = c / f["sma20"] - 1
    f["dist200"] = c / f["sma200"] - 1
    f["mom12_1"] = c.shift(21) / c.shift(252) - 1
    f["vol20"] = c.pct_change().rolling(20).std()
    f["hi252"] = c / df["High"].rolling(252).max() - 1
    f["liq"] = (c * df["Volume"]).rolling(60).median()     # traded value (local units)
    return f


def regime_series(mkt_df):
    """1 when the market index is above its 200-day average."""
    c = mkt_df["Close"]
    return (c > c.rolling(200).mean()).astype(float)


def add_rs_rank(feats):
    """Cross-sectional percentile of momentum, per market-agnostic calendar date."""
    mom = pd.DataFrame({s: f["mom"] for s, f in feats.items()})
    mom = mom.sort_index().ffill(limit=3)
    rank = mom.rank(axis=1, pct=True)
    for s, f in feats.items():
        f["rs"] = rank[s].reindex(f.index).fillna(0.5)
    return mom


def rs_from_distribution(mom_value, dist):
    """For on-demand tickers: percentile against the last saved universe distribution."""
    if mom_value is None or np.isnan(mom_value) or not dist:
        return 0.5
    return float(np.searchsorted(np.array(dist), mom_value)) / len(dist)


# ---------- scores ----------
def _ramp(x, a, b):
    """0 at a, 1 at b (works for a > b too)."""
    return np.clip((x - a) / (b - a), 0, 1)


def scores(f, regime):
    reg = regime.reindex(f.index).ffill().fillna(1.0)
    c = f["close"]
    comp_l = {
        "trend": 0.4 * (c > f["sma50"]) + 0.4 * (f["sma50"] > f["sma200"]) + 0.2 * (f["slope50"] > 0),
        "rs": f["rs"],
        "breakout": _ramp(c / f["hi20"] - 1, -0.05, 0.0),
        "rsi": np.minimum(_ramp(f["rsi"], 40, 50), _ramp(f["rsi"], 80, 70)),
        "volume": _ramp(f["volr"], 0.8, 2.0),
        "regime": reg,
    }
    comp_s = {
        "trend": 0.4 * (c < f["sma50"]) + 0.4 * (f["sma50"] < f["sma200"]) + 0.2 * (f["slope50"] < 0),
        "rs": 1 - f["rs"],
        "breakout": _ramp(c / f["lo20"] - 1, 0.05, 0.0),
        "rsi": np.minimum(_ramp(f["rsi"], 60, 50), _ramp(f["rsi"], 20, 30)),
        "volume": _ramp(f["volr"], 0.8, 2.0),
        "regime": 1 - reg,
    }
    f["score_long"] = 100 * sum(WEIGHTS[k] * comp_l[k] for k in WEIGHTS)
    f["score_short"] = 100 * sum(WEIGHTS[k] * comp_s[k] for k in WEIGHTS)
    valid = f["sma200"].notna()
    f.loc[~valid, ["score_long", "score_short"]] = np.nan
    dl = (f["score_long"] >= SIGNAL_MIN) & (f["score_long"] >= f["score_short"])
    ds = (f["score_short"] >= SIGNAL_MIN) & (f["score_short"] > f["score_long"])
    f["dir"] = np.select([dl, ds], [1, -1], 0)
    for k in WEIGHTS:
        f[f"cl_{k}"] = comp_l[k]
        f[f"cs_{k}"] = comp_s[k]
    return f


# ---------- trade simulation ----------
def simulate(df, atr_arr, direction, horizon):
    """Every day t is a potential signal: enter at open t+1, ATR stop/target, exit by close of day H.
    Returns (ret, outcome, exit_idx) arrays aligned to t; NaN where the trade isn't finished."""
    O, Hh, L, C = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    n = len(O)
    ret = np.full(n, np.nan)
    out = np.full(n, np.nan)
    xi = np.full(n, -1)
    t = np.arange(n - horizon)          # signals whose trade can finish
    if len(t) == 0:
        return ret, out, xi
    e = t + 1
    entry = O[e]
    a = atr_arr[t]
    if direction == 1:
        stop, tgt = entry - STOP_ATR * a, entry + TARGET_ATR * a
    else:
        stop, tgt = entry + STOP_ATR * a, entry - TARGET_ATR * a
    exitp = np.full(len(t), np.nan)
    oc = np.full(len(t), np.nan)
    ex = np.full(len(t), -1)
    alive = np.isfinite(a) & np.isfinite(entry)
    for k in range(horizon):
        d = e + k
        o, h, l, c = O[d], Hh[d], L[d], C[d]
        if direction == 1:
            gap_stop = alive & (k > 0) & (o <= stop)
            hit_stop = alive & ~gap_stop & (l <= stop)
            gap_tgt = alive & ~gap_stop & ~hit_stop & (k > 0) & (o >= tgt)
            hit_tgt = alive & ~gap_stop & ~hit_stop & ~gap_tgt & (h >= tgt)
        else:
            gap_stop = alive & (k > 0) & (o >= stop)
            hit_stop = alive & ~gap_stop & (h >= stop)
            gap_tgt = alive & ~gap_stop & ~hit_stop & (k > 0) & (o <= tgt)
            hit_tgt = alive & ~gap_stop & ~hit_stop & ~gap_tgt & (l <= tgt)
        for m, px, code in ((gap_stop, o, -1), (hit_stop, stop, -1), (gap_tgt, o, 1), (hit_tgt, tgt, 1)):
            exitp[m] = px[m] if np.ndim(px) else px
            oc[m] = code
            ex[m] = d[m]
            alive &= ~m
        if k == horizon - 1:
            exitp[alive] = c[alive]
            oc[alive] = 0
            ex[alive] = d[alive]
    if direction == 1:
        r = exitp / entry - 1 - COST
    else:
        r = (entry - exitp) / entry - COST
    ret[t], out[t], xi[t] = r, oc, ex
    return ret, out, xi


def hist_winrate(sig_mask, ret, exit_idx):
    """Win rate of this ticker's past signals in this direction, using only trades closed by day t."""
    n = len(ret)
    wins = np.zeros(n)
    cnt = np.zeros(n)
    # count independent trades only: a new signal counts once the previous counted trade has exited
    cand = np.flatnonzero(sig_mask & np.isfinite(ret) & (exit_idx >= 0))
    take, busy_until = [], -1
    for t in cand:
        if t > busy_until:
            take.append(t)
            busy_until = exit_idx[t]
    ok = np.zeros(n, bool)
    ok[take] = True
    np.add.at(cnt, exit_idx[ok], 1)
    np.add.at(wins, exit_idx[ok], (ret[ok] > 0).astype(float))
    c, w = np.cumsum(cnt), np.cumsum(wins)
    shr = (w + 0.5 * SHRINK_K) / (c + SHRINK_K)
    return w, c, shr


def market_trade(mkt_df, dates, horizon):
    """Buy-and-hold return of the market index for the same window (entry next open, exit day H close)."""
    m = mkt_df.reindex(mkt_df.index.union(dates)).sort_index()
    m[["Open", "Close"]] = m[["Open", "Close"]].bfill()
    O, C = m["Open"].to_numpy(), m["Close"].to_numpy()
    pos = m.index.get_indexer(dates)
    res = np.full(len(dates), np.nan)
    e, x = pos + 1, pos + horizon
    ok = x < len(O)
    res[ok] = C[x[ok]] / O[e[ok]] - 1
    return res


# ---------- presentation ----------
def levels(df, look=140, k=3):
    """Support/resistance from fractal pivots, clustered within 1.5%."""
    d = df.iloc[-look:]
    H, L = d["High"].to_numpy(), d["Low"].to_numpy()
    piv = []
    for i in range(2, len(d) - 2):
        if H[i] == H[i - 2:i + 3].max():
            piv.append(H[i])
        if L[i] == L[i - 2:i + 3].min():
            piv.append(L[i])
    piv.sort()
    clusters = []
    for p in piv:
        if clusters and p <= clusters[-1][-1] * 1.015:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    lv = [(float(np.mean(c)), len(c)) for c in clusters]
    last = float(d["Close"].iloc[-1])
    res = sorted([x for x in lv if x[0] > last], key=lambda x: x[0])[:k]
    sup = sorted([x for x in lv if x[0] <= last], key=lambda x: -x[0])[:k]
    return {"resistance": [round(x[0], 4) for x in res], "support": [round(x[0], 4) for x in sup]}


def reasons(row, direction, regime_name):
    """Hebrew explanation of the score — every line comes from a measured value."""
    r = []
    c, s50, s200 = row["close"], row["sma50"], row["sma200"]
    rs = row["rs"] * 100
    if direction >= 0:
        if c > s50 > s200:
            r.append(("+", "מגמה עולה: המחיר מעל ממוצע 50, שמעל ממוצע 200"))
        elif c > s200:
            r.append(("~", "מעל ממוצע 200, אבל המגמה הקצרה לא מאושרת"))
        else:
            r.append(("-", "מתחת לממוצע 200 — מגמה ארוכה שלילית"))
        r.append(("+" if rs >= 70 else "~" if rs >= 40 else "-", f"חוזק יחסי: חזקה מ-{rs:.0f}% מהנכסים שנסרקו"))
        gap = (c / row["hi20"] - 1) * 100 if row["hi20"] else 0
        if gap >= 0:
            r.append(("+", "פריצה מעל שיא 20 הימים"))
        else:
            r.append(("~" if gap > -5 else "-", f"{abs(gap):.1f}% מתחת לשיא 20 הימים"))
        rv = row["rsi"]
        r.append(("+" if 50 <= rv <= 70 else "-" if rv > 75 or rv < 40 else "~",
                  f"RSI {rv:.0f} — " + ("מומנטום בריא" if 50 <= rv <= 70 else "קנוי-יתר" if rv > 70 else "מומנטום חלש")))
    else:
        if c < s50 < s200:
            r.append(("+", "מגמה יורדת: המחיר מתחת לממוצע 50, שמתחת לממוצע 200"))
        else:
            r.append(("~", "מגמת ירידה חלקית"))
        r.append(("+" if rs <= 30 else "~", f"חולשה יחסית: חלשה מ-{100 - rs:.0f}% מהנכסים שנסרקו"))
        gap = (c / row["lo20"] - 1) * 100 if row["lo20"] else 0
        r.append(("+" if gap <= 0 else "~", "שבירה מתחת לשפל 20 הימים" if gap <= 0 else f"{gap:.1f}% מעל שפל 20 הימים"))
        rv = row["rsi"]
        r.append(("+" if 30 <= rv <= 50 else "~", f"RSI {rv:.0f}"))
    vr = row["volr"]
    r.append(("+" if vr >= 1.5 else "~", f"נפח מסחר פי {vr:.1f} מהממוצע"))
    up = row["cl_regime"] >= 0.5
    r.append(("+" if (up and direction >= 0) or (not up and direction < 0) else "-",
              f"השוק הכללי ({regime_name}) " + ("מעל " if up else "מתחת ל") + "ממוצע 200"))
    return [{"tone": t, "text": x} for t, x in r]
