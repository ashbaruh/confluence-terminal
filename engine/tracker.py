"""Live track record. Every pick is written before the next open and never edited afterwards,
except to fill in what actually happened. Random 'control' picks are tracked the same way."""
import json

import numpy as np
import pandas as pd

from config import COST, PICKS_PER_DAY, STATE, STOP_ATR, TARGET_ATR

FILE = STATE / "picks_history.json"


def load():
    if FILE.exists():
        return json.loads(FILE.read_text(encoding="utf-8"))
    return []


def save(hist):
    FILE.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")


def _mkt_ret(mdf, d0, d1):
    if mdf is None:
        return None
    try:
        o = mdf.loc[mdf.index >= pd.Timestamp(d0), "Open"].iloc[0]
        c = mdf.loc[mdf.index <= pd.Timestamp(d1), "Close"].iloc[-1]
        return float(c / o - 1)
    except Exception:
        return None


def update(hist, prices, mkt_prices):
    for p in hist:
        if p["status"] == "closed":
            continue
        df = prices.get(p["sym"])
        if df is None:
            continue
        bars = df[df.index > pd.Timestamp(p["signal_date"])]
        if bars.empty:
            p["status"] = "pending"
            continue
        d = p["direction"]
        entry = float(bars["Open"].iloc[0])
        a = p["atr"]
        stop = entry - d * STOP_ATR * a
        tgt = entry + d * TARGET_ATR * a
        p.update(entry=round(entry, 4), stop=round(stop, 4), target=round(tgt, 4),
                 entry_date=str(bars.index[0].date()))
        done = False
        for k in range(min(p["horizon"], len(bars))):
            o, h, l, c = (float(bars[x].iloc[k]) for x in ("Open", "High", "Low", "Close"))
            ex = oc = None
            if d == 1:
                if k > 0 and o <= stop: ex, oc = o, -1
                elif l <= stop: ex, oc = stop, -1
                elif k > 0 and o >= tgt: ex, oc = o, 1
                elif h >= tgt: ex, oc = tgt, 1
            else:
                if k > 0 and o >= stop: ex, oc = o, -1
                elif h >= stop: ex, oc = stop, -1
                elif k > 0 and o <= tgt: ex, oc = o, 1
                elif l <= tgt: ex, oc = tgt, 1
            if ex is None and k == p["horizon"] - 1:
                ex, oc = c, 0
            if ex is not None:
                r = (ex / entry - 1) if d == 1 else (entry - ex) / entry
                p.update(status="closed", outcome=oc, exit=round(ex, 4), exit_date=str(bars.index[k].date()),
                         ret=round(r - COST, 5), days=k + 1)
                m = _mkt_ret(mkt_prices.get(p["market"]) if p["market"] in mkt_prices else mkt_prices.get("US"),
                             p["entry_date"], p["exit_date"])
                p["market_ret"] = None if m is None else round(m, 5)
                done = True
                break
        if not done:
            last = float(bars["Close"].iloc[-1])
            r = (last / entry - 1) if d == 1 else (entry - last) / entry
            p.update(status="open", last=round(last, 4), unrealized=round(r - COST, 5), days=len(bars))
    return hist


def add_today(hist, picks, controls, run_date, horizon):
    if any(p.get("run_date") == run_date and p["kind"] == "pick" for p in hist):
        return hist  # already locked for this run date
    for kind, rows in (("pick", picks), ("control", controls)):
        for r in rows:
            hist.append({
                "id": f"{run_date}-{kind}-{r['sym']}", "kind": kind, "run_date": run_date,
                "signal_date": r["date"],
                "sym": r["sym"], "name": r["name"], "market": r["market"], "direction": int(r["direction"]),
                "score": round(float(r["score"]), 1), "win_rate": r.get("win_rate"), "cases": r.get("cases"),
                "horizon": int(horizon), "atr": float(r["atr"]), "status": "pending",
                "ref_close": round(float(r["close"]), 4),
            })
    return hist


def summary(hist):
    out = {}
    for kind in ("pick", "control"):
        closed = [p for p in hist if p["kind"] == kind and p["status"] == "closed"]
        r = np.array([p["ret"] for p in closed], float)
        m = np.array([p["market_ret"] for p in closed if p.get("market_ret") is not None], float)
        s = {"closed": len(closed), "open": sum(1 for p in hist if p["kind"] == kind and p["status"] != "closed")}
        if len(r):
            g, l = r[r > 0].sum(), -r[r < 0].sum()
            s.update(win_rate=float((r > 0).mean()), avg=float(r.mean()), total=float(r.sum()),
                     profit_factor=float(g / l) if l > 0 else None,
                     targets=sum(1 for p in closed if p["outcome"] == 1),
                     stops=sum(1 for p in closed if p["outcome"] == -1),
                     timeouts=sum(1 for p in closed if p["outcome"] == 0))
        if len(m):
            s["market_avg"] = float(m.mean())
        out[kind] = s
    return out
