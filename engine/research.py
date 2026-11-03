"""Which measurements actually predicted future returns? Run in GitHub Actions:
   python engine/research.py   → state/research.json

For each factor: daily cross-sectional rank correlation (IC) with the next 5/10-day return,
top-minus-bottom quintile spread, in-sample vs out-of-sample. Then a data-driven score
(weights fitted on in-sample ONLY) is backtested out-of-sample: top 5 a day vs random."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import backtest as bt
from config import OOS_FRACTION, PICKS_PER_DAY, RANDOM_SEEDS, STATE
from data import download
from main import clean
from pipeline import build, market_frames
from universe import build_universe

# factor name → (column, Hebrew label). Sign is NOT assumed; the data decides.
FACTORS = {
    "trend": ("cl_trend", "מגמה (מעל ממוצעים)"),
    "rs": ("rs", "חוזק יחסי 20/60 יום"),
    "breakout": ("cl_breakout", "קרבה לשיא 20 יום"),
    "rsi14": ("rsi", "RSI 14"),
    "volume": ("volr", "נפח מול ממוצע"),
    "rsi2": ("rsi2", "RSI 2 (קצר מאוד)"),
    "ret5": ("ret5", "תשואת 5 ימים אחרונים"),
    "ret1": ("ret1", "תשואת היום האחרון"),
    "dist20": ("dist20", "מרחק מממוצע 20"),
    "dist200": ("dist200", "מרחק מממוצע 200"),
    "mom12_1": ("mom12_1", "מומנטום 12 חודשים (בלי החודש האחרון)"),
    "vol20": ("vol20", "תנודתיות 20 יום"),
    "hi252": ("hi252", "מרחק משיא 52 שבועות"),
}
HS = [5, 10]


def fwd_returns(prices, panel):
    parts = []
    for s, df in prices.items():
        O, C = df["Open"], df["Close"]
        r = pd.DataFrame({"date": df.index, "sym": s})
        for h in HS:
            r[f"fwd{h}"] = (C.shift(-h) / O.shift(-1) - 1).to_numpy()
        parts.append(r)
    f = pd.concat(parts, ignore_index=True)
    return panel.merge(f, on=["date", "sym"], how="left")


def ranks(panel, col):
    return panel.groupby("date")[col].rank(pct=True)


def ic_series(panel, fcol, ycol):
    d = panel[["date", fcol, ycol]].dropna()
    d = d[d.groupby("date")[fcol].transform("size") >= 30]
    rx = d.groupby("date")[fcol].rank(pct=True)
    ry = d.groupby("date")[ycol].rank(pct=True)
    rx = rx - rx.groupby(d["date"]).transform("mean")
    ry = ry - ry.groupby(d["date"]).transform("mean")
    num = (rx * ry).groupby(d["date"]).sum()
    den = np.sqrt((rx ** 2).groupby(d["date"]).sum() * (ry ** 2).groupby(d["date"]).sum())
    return (num / den).dropna()


def summarize_ic(ic, h):
    if len(ic) < 10:
        return {"ic": None}
    sub = ic.iloc[::h]                      # non-overlapping dates for an honest t-stat
    t = sub.mean() / (sub.std() / np.sqrt(len(sub))) if sub.std() > 0 else 0
    return {"ic": float(ic.mean()), "t": float(t), "hit": float((ic > 0).mean()), "days": int(len(ic))}


def quintiles(panel, fcol, ycol):
    d = panel[["date", fcol, ycol]].dropna().copy()
    d["q"] = (d.groupby("date")[fcol].rank(pct=True) * 5).clip(upper=4.999).astype(int)
    m = d.groupby("q")[ycol].mean()
    return [float(m.get(i, np.nan)) for i in range(5)]


def main():
    uni = build_universe()
    mk, _ = market_frames(download)
    prices = download(list(uni))
    feats, panel, _ = build(prices, uni, mk)
    panel = fwd_returns(prices, panel)
    dates = np.sort(panel["date"].unique())
    split = dates[int(len(dates) * (1 - OOS_FRACTION))]
    is_p, oos_p = panel[panel["date"] < split], panel[panel["date"] >= split]

    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "start": str(pd.Timestamp(dates[0]).date()), "split": str(pd.Timestamp(split).date()),
           "end": str(pd.Timestamp(dates[-1]).date()), "rows": int(len(panel)), "factors": {}}

    # 1) each factor alone
    for name, (col, label) in FACTORS.items():
        r = {"label": label}
        for h in HS:
            y = f"fwd{h}"
            r[f"is_{h}"] = summarize_ic(ic_series(is_p, col, y), h)
            r[f"oos_{h}"] = summarize_ic(ic_series(oos_p, col, y), h)
            r[f"oos_q_{h}"] = quintiles(oos_p, col, y)
        out["factors"][name] = r
        print(f"{name:9s} IS ic10={r['is_10'].get('ic')}  OOS ic10={r['oos_10'].get('ic')}  t={r['oos_10'].get('t')}")

    # market regime (time-series, not cross-sectional)
    reg = {}
    for h in HS:
        g = oos_p.groupby(oos_p["cl_regime"] >= 0.5)[f"fwd{h}"].mean()
        reg[str(h)] = {"above_200": float(g.get(True, np.nan)), "below_200": float(g.get(False, np.nan))}
    out["regime_oos"] = reg

    # 2) data-driven score: weights = in-sample IC (10d), only factors with |t| >= 2 in-sample
    w = {}
    for name, (col, _) in FACTORS.items():
        s = out["factors"][name]["is_10"]
        if s.get("ic") is not None and abs(s["t"]) >= 2:
            w[name] = s["ic"]
    out["weights"] = w
    print("weights (in-sample):", w)
    if w:
        comp = sum(wt * (ranks(panel, FACTORS[n][0]) - 0.5) for n, wt in w.items())
        panel["dd_score"] = comp
        panel["dd_score"] = ranks(panel, "dd_score") * 100
        res = {}
        rng = np.random.default_rng(1)
        for h in HS:
            for kind, col in (("trade", f"ret_l{h}"), ("raw", f"fwd{h}")):
                v = panel[np.isfinite(panel[col])]
                picks = bt.pick_no_overlap(v, "dd_score", f"xd_l{h}", PICKS_PER_DAY)
                pool = v[v["date"].isin(picks["date"].unique())]
                for per, sel in (("is", picks["date"] < split), ("oos", picks["date"] >= split)):
                    p = picks[sel]
                    pl = pool[pool["date"] < split] if per == "is" else pool[pool["date"] >= split]
                    rnd = []
                    for _ in range(RANDOM_SEEDS):
                        rp = pl.assign(_k=rng.random(len(pl))).sort_values(["date", "_k"]).groupby("date").head(PICKS_PER_DAY)
                        rnd.append(rp[col].mean())
                    st = bt.stats(p[col])
                    st.update(random_avg=float(np.mean(rnd)), edge=float(st.get("avg", 0) - np.mean(rnd)),
                              p_value=float(np.mean(np.array(rnd) >= st.get("avg", 0))),
                              market_avg=float(np.nanmean(p[f"mkt{h}"])))
                    res[f"{kind}_{h}_{per}"] = st
                    print(kind, h, per, {k: round(x, 4) if isinstance(x, float) else x for k, x in st.items()
                                          if k in ("n", "win_rate", "avg", "random_avg", "edge", "p_value", "market_avg")})
        # score buckets OOS: does a higher score mean a better outcome?
        o = panel[panel["date"] >= split]
        bk = (o["dd_score"] // 10).clip(upper=9)
        res["oos_buckets_10"] = {f"{int(b)*10}-{int(b)*10+9}": {"fwd10": float(g["fwd10"].mean()),
                                                                  "win": float((g["fwd10"] > 0).mean()), "n": int(len(g))}
                                 for b, g in o.groupby(bk)}
        out["dd_backtest"] = res
    (STATE / "research.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
