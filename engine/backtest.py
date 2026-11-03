"""Honest backtest: daily top-5 vs random entries (same exits) vs the market index."""
import numpy as np
import pandas as pd

from config import DEFAULT_HORIZON, HORIZONS, OOS_FRACTION, PICKS_PER_DAY, RANDOM_SEEDS


def rank_value(score, wr):
    return 0.5 * score / 100 + 0.5 * wr


def stats(r, out=None):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"n": 0}
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    s = {
        "n": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg": float(r.mean()),
        "median": float(np.median(r)),
        "profit_factor": float(g / l) if l > 0 else None,
        "best": float(r.max()),
        "worst": float(r.min()),
    }
    if out is not None:
        o = np.asarray(out, float)
        o = o[np.isfinite(o)]
        if len(o):
            s["target_pct"] = float((o == 1).mean())
            s["stop_pct"] = float((o == -1).mean())
            s["timeout_pct"] = float((o == 0).mean())
    return s


def top_n(df, key, n):
    return df.sort_values(["date", key], ascending=[True, False]).groupby("date", sort=False).head(n)


def pick_no_overlap(sig, rank_col, exit_col, n):
    """Daily top-n, skipping symbols still held from an earlier pick (no double counting)."""
    sig = sig.sort_values(["date", rank_col], ascending=[True, False])
    held = {}
    keep = []
    for date, g in sig.groupby("date", sort=True):
        k = 0
        for i, s, xd in zip(g.index, g["sym"], g[exit_col]):
            if s in held and held[s] >= date:
                continue
            keep.append(i)
            held[s] = xd
            k += 1
            if k == n:
                break
    return sig.loc[keep]


def run(panel):
    dates = np.sort(panel["date"].unique())
    if len(dates) < 50:
        return {"error": "not enough history"}
    split = dates[int(len(dates) * (1 - OOS_FRACTION))]
    res = {"split_date": str(pd.Timestamp(split).date()), "start": str(pd.Timestamp(dates[0]).date()),
           "end": str(pd.Timestamp(dates[-1]).date()), "horizons": {}}
    rng_master = np.random.default_rng(42)
    for h in HORIZONS:
        hres = {}
        for d, tag in ((1, "l"), (-1, "s")):
            col, oc, wr = f"ret_{tag}{h}", f"out_{tag}{h}", f"wr_{tag}{h}"
            sc = "score_long" if d == 1 else "score_short"
            valid = panel[np.isfinite(panel[col])]
            sig = valid[valid["dir"] == d].copy()
            sig["rank"] = rank_value(sig[sc], sig[wr])
            picks = pick_no_overlap(sig, "rank", f"xd_{tag}{h}", PICKS_PER_DAY)
            pdays = picks["date"].unique()
            pool = valid[valid["date"].isin(pdays)]
            # random benchmark on the same days, same direction, same exits
            seeds_avg, seeds_is, seeds_oos = [], [], []
            rand_daily = None
            for k in range(RANDOM_SEEDS):
                key = rng_master.random(len(pool))
                rp = pool.assign(_k=key).sort_values(["date", "_k"]).groupby("date", sort=False).head(PICKS_PER_DAY)
                seeds_avg.append(rp[col].mean())
                seeds_is.append(rp.loc[rp["date"] < split, col].mean())
                seeds_oos.append(rp.loc[rp["date"] >= split, col].mean())
                dm = rp.groupby("date")[col].mean()
                rand_daily = dm if rand_daily is None else rand_daily.add(dm, fill_value=0)
            rand_daily = rand_daily / RANDOM_SEEDS

            def block(p, rs_list):
                s = stats(p[col], p[oc])
                if s["n"]:
                    ra = float(np.nanmean(rs_list))
                    s["random_avg"] = ra
                    s["edge"] = s["avg"] - ra
                    s["p_value"] = float(np.mean(np.array(rs_list) >= s["avg"]))
                    if d == 1:
                        s["market_avg"] = float(np.nanmean(p[f"mkt{h}"]))
                return s

            is_p, oos_p = picks[picks["date"] < split], picks[picks["date"] >= split]
            r = {"all": block(picks, seeds_avg), "in_sample": block(is_p, seeds_is),
                 "out_of_sample": block(oos_p, seeds_oos)}
            # per year
            yr = {}
            for y, g in picks.groupby(picks["date"].dt.year):
                st = stats(g[col])
                yr[str(y)] = {"n": st["n"], "win_rate": st.get("win_rate"), "avg": st.get("avg")}
            r["by_year"] = yr
            # per market
            r["by_market"] = {m: stats(g[col]) for m, g in picks.groupby("market")}
            # equity: each day 1/H of capital goes into that day's picks
            day_ret = picks.groupby("date")[col].mean()
            idx = pd.DatetimeIndex(sorted(set(day_ret.index) | set(rand_daily.index)))
            eq = (1 + day_ret.reindex(idx).fillna(0) / h).cumprod()
            eqr = (1 + rand_daily.reindex(idx).fillna(0) / h).cumprod()
            curve = {"dates": [str(x.date()) for x in idx], "strategy": eq.round(4).tolist(),
                     "random": eqr.round(4).tolist()}
            if d == 1:
                mk = picks.groupby("date")[f"mkt{h}"].mean().reindex(idx).fillna(0)
                curve["market"] = (1 + mk / h).cumprod().round(4).tolist()
            step = max(1, len(idx) // 300)
            r["equity"] = {k: v[::step] for k, v in curve.items()}
            hres["long" if d == 1 else "short"] = r
        res["horizons"][str(h)] = hres

    # decisions made by the data, not by hand
    def oos(h, side):
        return res["horizons"][str(h)][side]["out_of_sample"]

    cands = [h for h in HORIZONS if oos(h, "long").get("n", 0) >= 30 and oos(h, "long").get("edge", -1) > 0
             and oos(h, "long").get("avg", -1) > 0]
    best = max(cands, key=lambda h: oos(h, "long")["avg"]) if cands else DEFAULT_HORIZON
    so = oos(best, "short")
    res["decision"] = {
        "horizon": int(best),
        "horizon_reason": "best out-of-sample average with positive edge over random" if cands
        else "no horizon beat random out-of-sample — using default, treat signals with caution",
        "long_edge": bool(cands),
        "shorts_enabled": bool(so.get("n", 0) >= 30 and so.get("edge", -1) > 0 and so.get("avg", -1) > 0),
    }
    return res
