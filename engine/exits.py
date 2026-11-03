"""Can we win 4 out of 5? Exit-rule study on Tel Aviv, using the data-driven score from research_il.json.
For the daily top-5 (and random picks), try: take profit at +X% any time within the window,
optional stop at -Y%, otherwise exit at the close of day 20. Output: state/exits_il.json"""
import json

import numpy as np
import pandas as pd

from config import OOS_FRACTION, PICKS_PER_DAY, STATE
from data import download
from main import clean
from pipeline import build, market_frames
from universe import tase_all

HOLD = 20
COST = 0.004                    # Tel Aviv round trip incl. spread (conservative)
TARGETS = [0.01, 0.02, 0.03, 0.05, None]
STOPS = [None, 0.10, 0.15]
MIN_LIQ = 2e7
FACT = {"mom12_1": "mom12_1", "vol20": "vol20", "hi252": "hi252", "dist200": "dist200",
        "trend": "cl_trend", "rs": "rs", "volume": "volr"}


def sim(df_arr, i, tgt, stp):
    """Enter next open after day i. Returns (return, hit_target) or None if window incomplete/glitched."""
    O, H, L, C = df_arr
    e = i + 1
    if e + HOLD - 1 >= len(O):
        return None
    win = C[e - 1:e + HOLD]
    if np.any(np.abs(np.diff(win) / win[:-1]) > 0.35):
        return None
    entry = O[e]
    for k in range(HOLD):
        d = e + k
        if stp is not None:
            sp = entry * (1 - stp)
            if k > 0 and O[d] <= sp:
                return O[d] / entry - 1 - COST, False
            if L[d] <= sp:
                return -stp - COST, False
        if tgt is not None:
            tp = entry * (1 + tgt)
            if k > 0 and O[d] >= tp:
                return O[d] / entry - 1 - COST, True
            if H[d] >= tp:
                return tgt - COST, True
    return C[e + HOLD - 1] / entry - 1 - COST, False


def summ(r):
    r = np.array([x for x in r if x is not None and np.isfinite(x)])
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "win_rate": float((r > 0).mean()), "avg": float(r.mean()),
            "median": float(np.median(r)), "worst": float(r.min()), "p5": float(np.percentile(r, 5)),
            "avg_win": float(r[r > 0].mean()) if (r > 0).any() else None,
            "avg_loss": float(r[r <= 0].mean()) if (r <= 0).any() else None}


def main():
    w = json.loads((STATE / "research_il.json").read_text(encoding="utf-8"))["weights"]
    uni = tase_all()
    mk, _ = market_frames(download)
    prices = download(list(uni))
    _, panel, _ = build(prices, uni, mk)
    panel = panel[panel["liq"] >= MIN_LIQ].copy()
    comp = sum(wt * (panel.groupby("date")[FACT[n]].rank(pct=True) - 0.5) for n, wt in w.items())
    panel["score"] = comp
    panel["score"] = panel.groupby("date")["score"].rank(pct=True) * 100
    dates = np.sort(panel["date"].unique())
    split = dates[int(len(dates) * (1 - OOS_FRACTION))]
    oos = panel[panel["date"] >= split]
    arrs = {s: tuple(df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close")) for s, df in prices.items()}
    pos = {s: pd.Index(df.index) for s, df in prices.items()}

    # daily top-5, a symbol is not re-picked while held (HOLD days)
    picks, held = [], {}
    for d, g in oos.sort_values(["date", "score"], ascending=[True, False]).groupby("date", sort=True):
        k = 0
        for s, sc in zip(g["sym"], g["score"]):
            i = pos[s].get_loc(d)
            if s in held and held[s] >= i:
                continue
            picks.append((s, i, sc))
            held[s] = i + HOLD
            k += 1
            if k == PICKS_PER_DAY:
                break
    rng = np.random.default_rng(3)
    rnd = []
    for d, g in oos.groupby("date"):
        for s in rng.choice(g["sym"].to_numpy(), size=min(PICKS_PER_DAY, len(g)), replace=False):
            rnd.append((s, pos[s].get_loc(d), None))
    # also: only the very best scores (>= 95)
    top = [p for p in picks if p[2] >= 95]

    out = {"split": str(pd.Timestamp(split).date()), "hold": HOLD, "cost": COST, "weights": w, "schemes": []}
    for tgt in TARGETS:
        for stp in STOPS:
            row = {"target": tgt, "stop": stp}
            for name, lst in (("picks", picks), ("top95", top), ("random", rnd)):
                res = [sim(arrs[s], i, tgt, stp) for s, i, _ in lst]
                res = [r for r in res if r is not None]
                st = summ([r[0] for r in res])
                st["target_hit"] = float(np.mean([r[1] for r in res])) if res else None
                row[name] = st
            row["edge"] = row["picks"].get("avg", 0) - row["random"].get("avg", 0)
            out["schemes"].append(row)
            p, r = row["picks"], row["random"]
            print(f"tgt={tgt} stop={stp}: picks win {p.get('win_rate',0):.2f} avg {p.get('avg',0)*100:+.2f}% "
                  f"worst {p.get('worst',0)*100:.1f}% | random win {r.get('win_rate',0):.2f} avg {r.get('avg',0)*100:+.2f}%")
    (STATE / "exits_il.json").write_text(json.dumps(clean(out), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
