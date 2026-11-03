"""Confluence Terminal.
  python engine/main.py daily          full scan, backtest, picks, track record, site data
  python engine/main.py ticker NVDA    analyse one extra symbol and add it to the site
"""
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import backtest as bt
import tracker
from config import (CHART_BARS, DEFAULT_HORIZON, HORIZONS, MARKETS, PICKS_PER_DAY, SITE_DATA, STATE,
                    WEIGHTS)
from data import SYNTHETIC, download
from model import (features, hist_winrate, levels, reasons, regime_series, rs_from_distribution, scores,
                   simulate)
from pipeline import build, market_frames
from universe import build_universe, load_extra, market_of

REGIME_NAME = {"US": "S&P 500", "IL": "ת\"א 125", "CRYPTO": "ביטקוין"}


def clean(o):
    """JSON-safe: NaN → None, numpy → python, rounded floats."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o.date())
    return o


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def fname(sym):
    return sym.replace("^", "_").replace("/", "_") + ".json"


def analysis(sym, info, df, f, horizon):
    """Everything the ticker page and the pick card need, from the last row."""
    row = f.iloc[-1]
    d = int(row["dir"])
    tag = "s" if d == -1 else "l"
    sc = float(row["score_short"] if d == -1 else row["score_long"])
    side = -1 if d == -1 else 1
    a = float(row["atr"])
    c = float(row["close"])
    comps = {k: float(row[f"{'cs' if d == -1 else 'cl'}_{k}"]) for k in WEIGHTS}
    hz = {}
    for h in HORIZONS:
        if f"wn_{tag}{h}" in f:
            hz[str(h)] = {"wins": int(row[f"wn_{tag}{h}"]), "cases": int(row[f"n_{tag}{h}"]),
                          "win_rate": float(row[f"wn_{tag}{h}"] / row[f"n_{tag}{h}"]) if row[f"n_{tag}{h}"] else None}
    main = hz.get(str(horizon), {})
    return {
        "sym": sym, "name": info.get("name", sym), "market": info.get("market", market_of(sym)),
        "sector": info.get("sector", ""), "date": str(f.index[-1].date()), "close": c,
        "chg1d": float(c / f["close"].iloc[-2] - 1) if len(f) > 1 else None,
        "direction": d, "score": sc, "score_long": float(row["score_long"]), "score_short": float(row["score_short"]),
        "components": comps, "win_rate": main.get("win_rate"), "cases": main.get("cases", 0),
        "wr_rank": float(row.get(f"wr_{tag}{horizon}", 0.5)), "by_horizon": hz, "horizon": horizon,
        "plan": {"ref": c, "stop": c - side * 1.5 * a, "target": c + side * 3.0 * a, "atr": a,
                 "stop_pct": -1.5 * a / c, "target_pct": 3.0 * a / c},
        "rsi": float(row["rsi"]), "rs": float(row["rs"]), "volr": float(row["volr"]), "atr": a,
        "reasons": reasons(row, d, REGIME_NAME.get(info.get("market", "US"), "השוק")),
    }


def ticker_file(sym, an, df, f):
    tail = df.iloc[-CHART_BARS:]
    ft = f.iloc[-CHART_BARS:]
    marks = []
    prev = ft["dir"].shift(1).fillna(0)
    for dt, v, pv in zip(ft.index, ft["dir"], prev):
        if v != 0 and v != pv:
            marks.append({"t": str(dt.date()), "d": int(v)})
    return {
        **an,
        "bars": [[str(i.date()), r.Open, r.High, r.Low, r.Close, r.Volume] for i, r in tail.iterrows()],
        "sma50": ft["sma50"].tolist(), "sma200": ft["sma200"].tolist(),
        "levels": levels(df), "signals": marks[-40:],
    }


def screener_row(an):
    return {k: an[k] for k in ("sym", "name", "market", "sector", "close", "chg1d", "direction", "score",
                               "score_long", "score_short", "win_rate", "cases", "rsi", "rs", "date")}


def daily():
    run_date = datetime.now(timezone.utc).date().isoformat()
    uni = build_universe(include_us=True)
    mk, _ = market_frames(download)
    prices = download(list(uni))
    print("building panel…")
    feats, panel, mom = build(prices, uni, mk)
    print("backtest…")
    res = bt.run(panel)
    dec = res.get("decision", {"horizon": DEFAULT_HORIZON, "shorts_enabled": False, "long_edge": False})
    H = dec["horizon"]
    dump(STATE / "backtest.json", res)
    dump(SITE_DATA / "backtest.json", res)

    # latest analysis for every asset
    last_date = max(f.index[-1] for f in feats.values())
    analyses = {}
    for s, f in feats.items():
        if f["score_long"].notna().iloc[-1] and (last_date - f.index[-1]).days <= 5:
            an = analysis(s, uni.get(s, {}), prices[s], f, H)
            analyses[s] = an
            dump(SITE_DATA / "t" / fname(s), ticker_file(s, an, prices[s], f))

    hist = tracker.update(tracker.load(), prices, mk)
    held = {p["sym"] for p in hist if p["kind"] == "pick" and p["status"] != "closed"}

    allowed = {1} | ({-1} if dec["shorts_enabled"] else set())
    cands = [a for a in analyses.values() if a["direction"] in allowed and a["sym"] not in held]
    for a in cands:
        a["rank"] = bt.rank_value(a["score"], a["wr_rank"])
    picks = sorted(cands, key=lambda a: -a["rank"])[:PICKS_PER_DAY]
    shorts_watch = sorted([a for a in analyses.values() if a["direction"] == -1],
                          key=lambda a: -a["score"])[:PICKS_PER_DAY] if not dec["shorts_enabled"] else []

    rng = np.random.default_rng(int(run_date.replace("-", "")))
    pool = list(analyses.values())
    controls = [dict(a, direction=1) for a in rng.choice(pool, size=min(PICKS_PER_DAY, len(pool)), replace=False)] if pool else []

    hist = tracker.add_today(hist, picks, controls, run_date, H)
    tracker.save(hist)

    regime = {}
    for m, df in mk.items():
        r = regime_series(df)
        regime[m] = {"above_200": bool(r.iloc[-1]), "close": float(df["Close"].iloc[-1]),
                     "chg1d": float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1), "name": REGIME_NAME[m]}

    latest_mom = mom.ffill().iloc[-1].dropna().sort_values().to_numpy()
    dist = np.quantile(latest_mom, np.linspace(0, 1, 201)).tolist() if len(latest_mom) else []
    dump(SITE_DATA / "today.json", {"run_date": run_date, "data_date": str(last_date.date()), "horizon": H,
                                    "picks": picks, "shorts_watch": shorts_watch})
    dump(SITE_DATA / "screener.json", [screener_row(a) for a in analyses.values()])
    dump(SITE_DATA / "history.json", {"picks": hist[-1500:], "summary": tracker.summary(hist)})
    dump(SITE_DATA / "meta.json", {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                                   "run_date": run_date, "data_date": str(last_date.date()),
                                   "synthetic": SYNTHETIC, "assets": len(analyses), "decision": dec,
                                   "regime": regime, "mom_dist": dist})
    print(f"done: {len(analyses)} assets, {len(picks)} picks, horizon {H}")


def ticker(sym):
    import re
    sym = sym.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-^=]{0,14}", sym):
        print("invalid symbol:", sym)
        sys.exit(1)
    meta_p = SITE_DATA / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    H = meta.get("decision", {}).get("horizon", DEFAULT_HORIZON)
    mkt = market_of(sym)
    got = download([sym, MARKETS[mkt], MARKETS["US"]])
    if sym not in got:
        dump(SITE_DATA / "t" / fname(sym), {"sym": sym, "error": "לא נמצאו נתונים לסימול הזה ב-Yahoo Finance"})
        print("no data for", sym)
        return
    df = got[sym]
    f = features(df)
    dist = meta.get("mom_dist", [])
    f["rs"] = [rs_from_distribution(v, dist) for v in f["mom"]]
    mdf = got.get(MARKETS[mkt]) if MARKETS[mkt] in got else got.get(MARKETS["US"])
    scores(f, regime_series(mdf) if mdf is not None else pd.Series(1.0, index=f.index))
    for h in HORIZONS:
        for d, tag in ((1, "l"), (-1, "s")):
            ret, out, xi = simulate(df, f["atr"].to_numpy(), d, h)
            w, c, shr = hist_winrate(f["dir"].to_numpy() == d, ret, xi)
            f[f"wn_{tag}{h}"], f[f"n_{tag}{h}"], f[f"wr_{tag}{h}"] = w, c, shr
    info = {"name": sym, "market": mkt, "sector": "Custom"}
    an = analysis(sym, info, df, f, H)
    dump(SITE_DATA / "t" / fname(sym), ticker_file(sym, an, df, f))
    sp = SITE_DATA / "screener.json"
    rows = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else []
    rows = [r for r in rows if r["sym"] != sym] + [screener_row(an)]
    dump(sp, rows)
    extra = load_extra()
    if sym not in extra:
        extra.append(sym)
        (STATE / "extra_tickers.json").write_text(json.dumps(extra), encoding="utf-8")
    print("analysed", sym, an["direction"], round(an["score"], 1))


if __name__ == "__main__":
    STATE.mkdir(exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if cmd == "ticker":
        ticker(sys.argv[2])
    else:
        daily()
