"""Rolling-origin backtest for the gta6-trophy-rarity-forecast-01 entry.

Two parts. First, a walk-forward over the kit's practice histories: those pages
never changed, so every model that is honest about it collapses to persistence and
the two tie; it is run for the record, not for signal. Second, a walk-forward over
synthetic week-one histories that carry a trend: players tracked and every trophy's
own count grow and bend, reads every ten minutes on the recorder's clock, the page
only changing at Exophase's refreshes, rates printed to two decimals like the real
page does. Origins at 24, 48 and 60 hours after the first read, targets 24 to 108
hours beyond that, so the horizons include the final's 4.5 days.

Three estimators are compared on the contest's own loss: the mean pinball loss on
the logit scale, 2*max(tau*d, (tau-1)*d) with d = logit(truth) - logit(q), averaged
over trophy and quantile. Persistence is the house baseline the prize bar sits on;
the curve baseline is the house's straight line in ln(1 + hours); the model is this
repository's main.py. Empirical coverage of the 90 percent band is reported too, since
a band that does not cover tells you the spread is wrong even when the loss looks
close.

Standard library only, deterministic: the same seed list gives the same table.
"""

import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as M

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = M.Z
# Anchors read off the case files: the first recorder's read at 2026-11-18T23:50Z
# showed 638 players tracked, and the page had refreshed by the 04:30 read to 3868.
N_FIRST = 638.0
U_REFRESH, N_REFRESH = 5.3, 3868.0
U_END = 168.0  # hours, the end of week one


def logit(pct):
    return M.logit(pct)


def pinball(quantiles, truth_pct):
    """Mean pinball loss on the logit scale over the five quantiles."""
    t = logit(truth_pct)
    return sum(2.0 * max(tau * (t - logit(q)), (tau - 1.0) * (t - logit(q)))
               for tau, q in zip(LEVELS, quantiles)) / len(LEVELS)


def _integral(T, s, u):
    """The integral of 1/(1+t/T)^s from 0 to u: T*ln(1+u/T) when s is 1."""
    if abs(s - 1.0) < 1e-9:
        return T * math.log1p(u / T)
    return T / (1.0 - s) * ((1.0 + u / T) ** (1.0 - s) - 1.0)


def _pin_g0(base, target, u, T, s):
    """The g0 that makes base * exp(g0 * integral) equal target at u."""
    return math.log(target / base) / _integral(T, s, u)


def _curve(base, g0, T, s):
    """u -> base * exp(g0 * integral from 0 to u of 1/(1+t/T)^s dt).

    s near 1 is a pure power law, a straight line in ln-ln; s above 1 bends it down
    toward a ceiling, which is how week-one growth actually behaves."""
    def f(u):
        return base * math.exp(g0 * _integral(T, s, u))
    return f


def _pin_curve(base, target, u, T, s):
    return _curve(base, _pin_g0(base, target, u, T, s), T, s)


def _draw(rng, lo, hi):
    """A log-uniform draw, which spreads the sampling over orders of magnitude."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _launch_night_rates(rng):
    """The launch-night page, from the case files: the platinum under a percent, the
    trophy handed out at game start at a hundred, the rest in a narrow band."""
    rates = [0.63] + [100.0] * 54
    for i in range(2, 55):
        rates[i] = round(rng.uniform(0.55, 2.30), 2)
    return rates


def world(seed):
    """One plausible week one, drawn until it agrees with what the page actually did:
    every rate but the auto trophy's lower at 5.3 hours, the platinum the hardest hit,
    a tracked count that ends the week somewhere between 30k and 3m, and week-one rates
    still under a quarter. Resampling is the filter; the draws themselves are wide."""
    rng = random.Random(seed)
    n_rates = _launch_night_rates(rng)
    for _ in range(4000):
        T_n = _draw(rng, 18.0, 140.0)
        s_n = rng.uniform(0.60, 2.30)
        N = _pin_curve(N_FIRST, N_REFRESH, U_REFRESH, T_n, s_n)
        rates = _launch_night_rates(rng)
        trophies = []
        for i in range(55):
            e0 = N_FIRST * rates[i] / 100.0
            g0 = _draw(rng, 0.04, 0.55)
            T_e = _draw(rng, 5.0, 90.0)
            s_e = rng.uniform(0.55, 2.60)
            e = _curve(e0, g0, T_e, s_e)
            trophies.append({"id": i + 1, "e0": e0, "g0": g0, "T": T_e, "s": s_e, "e": e})
        r0 = [100.0 * t["e"](0.0) / N(0.0) for t in trophies]
        r5 = [100.0 * t["e"](U_REFRESH) / N(U_REFRESH) for t in trophies]
        r_end = [100.0 * t["e"](U_END) / N(U_END) for t in trophies]
        platinum = r5[0] < r0[0]
        dropped = sum(1 for i in range(2, 55) if r5[i] < r0[i])
        if not (platinum and dropped >= 48):
            continue
        if not (30000.0 <= N(U_END) <= 3000000.0):
            continue
        if not all(0.003 <= r <= 25.0 for r in r_end):
            continue
        break
    return {"N": N, "trophies": trophies, "rates0": rates}


def page_at(w, u):
    """The page's own numbers at time u: tracked players, total earned, rates printed
    to two decimals like the real page does."""
    N = w["N"](u)
    E = 0.0
    rates = []
    for tr in w["trophies"]:
        e = tr["e"](u)
        E += e
        rates.append(round(100.0 * e / N, 2))
    return int(round(N)), int(round(E)), rates


def _refresh_at(u, refresh):
    """The page shows the state at the last refresh at or before u."""
    if u <= 0.0:
        return 0.0
    return math.floor(u / refresh + 1e-9) * refresh


def history(w, u_origin, refresh):
    """Reads on the recorder's clock up to u_origin: every half hour until the in
    cluster recorder starts, every ten minutes after."""
    rows = []
    u, step = 0.0, 0.5
    while u < u_origin - 1e-9:
        N, E, rates = page_at(w, _refresh_at(u, refresh))
        rows.append({"slot": u, "players_tracked": N, "trophies_earned": E, "rates": rates})
        u = min(u + step, u_origin)
        if u >= 12.0 - 1e-9 and step > 0.16667:
            step = 1.0 / 6.0
    return rows


def make_case(w, u_origin, u_target, refresh):
    """One synthetic case: the reads up to the origin, a target beyond it, and the
    rates the page prints at the target, which are what the contest scores."""
    rows = history(w, u_origin, refresh)
    if len(rows) < 8:
        return None
    at = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
    for r in rows:
        r["at"] = (at + timedelta(hours=r["slot"])).isoformat().replace("+00:00", "Z")
    _, _, truth = page_at(w, _refresh_at(u_target, refresh))
    trophies = [{"id": t["id"], "name": "t%d" % t["id"],
                 "description": "", "tier": "bronze", "points": 15, "secret": False}
                for t in w["trophies"]]
    case = {
        "schema": "makefun/gta6-trophies-case/v1", "id": "sim",
        "at": rows[-1]["at"],
        "horizon_minutes": round((u_target - u_origin) * 60.0, 1),
        "target": (at + timedelta(hours=u_target)).isoformat().replace("+00:00", "Z"),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19", "source": "exophase"},
        "trophies": trophies, "history": rows, "notes": "synthetic week one",
    }
    return case, truth


def house_curve(case):
    """House curve baseline: a least squares line of logit rate on ln(1 + hours since
    the first read), fitted on the last 24 hours, read at the target; spread is the
    residual spread plus a random walk's."""
    tl, first = M._build_timeline(case)
    if tl is None:
        return None
    target_hours = (M.parse_ts(case["target"]) - first).total_seconds() / 3600.0
    x_t = math.log1p(target_hours)
    out = []
    for i in range(len(case["trophies"])):
        xs, ys, hs = [], [], []
        for h, _, _, rates in tl:
            if rates is None or i >= len(rates) or not isinstance(rates[i], (int, float)):
                continue
            if h >= tl[-1][0] - 24.0 and rates[i] >= 0.01:
                xs.append(math.log1p(h))
                ys.append(logit(rates[i]))
                hs.append(h)
        if len(xs) < 3:
            out.append(None)
            continue
        res = M.wls([[x] for x in xs], ys)
        if res[0] is None:
            out.append(None)
            continue
        a, b = res[0]
        resid = [ys[k] - (a + b * xs[k]) for k in range(len(xs))]
        sd_r = math.sqrt(sum(r * r for r in resid) / max(1, len(xs) - 2)) if len(xs) > 2 else 0.0
        # Residual spread plus a random walk's over the days from the last read.
        days = max(0.0, target_hours - max(hs)) / 24.0
        sd = math.sqrt(max(sd_r, 0.0) + 0.25 * days)
        out.append((a + b * x_t, sd))
    return out


def mine(case):
    est = M.estimate(case)
    return None if est is None else est


def run(seed_lo, seed_hi, origins=(24.0, 48.0, 60.0), horizons=(24.0, 48.0, 72.0, 96.0, 108.0),
        sweeps=None):
    rows = []
    for seed in range(seed_lo, seed_hi):
        w = world(seed)
        for refresh in (0.5, 1.5):
            for u_origin in origins:
                for dh in horizons:
                    u_target = u_origin + dh
                    got = make_case(w, u_origin, u_target, refresh)
                    if got is None:
                        continue
                    case, truth = got
                    res = {"seed": seed, "origin": u_origin, "dh": dh, "refresh": refresh}
                    for name, est in (("persist", None), ("curve", house_curve(case)), ("model", mine(case))):
                        qs = []
                        n = 0
                        for i, t in enumerate(case["trophies"]):
                            if est is not None and i < len(est) and est[i] is not None:
                                med, sd = est[i]
                                q = [M.expit(med + z * sd) for z in Z]
                            else:
                                q = M.persistence_quantiles(
                                    case["history"][-1]["rates"][i], case["horizon_minutes"])
                            qs.append(q)
                            n += 1
                            res.setdefault(f"cover90_{name}", []).append(
                                (logit(truth[i]) >= logit(q[0])) and (logit(truth[i]) <= logit(q[-1])))
                        res[f"loss_{name}"] = sum(pinball(q, truth[i]) for i, q in enumerate(qs)) / n
                    rows.append(res)
    return rows


def summarise(rows, tag=""):
    if not rows:
        print("  (no rows)")
        return
    for name in ("persist", "curve", "model"):
        vals = [r[f"loss_{name}"] for r in rows]
        cov = [sum(c) / len(c) for c in (r.get(f"cover90_{name}", [0]) for r in rows) if c]
        wins = sum(1 for r in rows
                   if r[f"loss_{name}"] < r[f"loss_persist"] - 1e-12) if name != "persist" else 0
        print(f"  {name:8s} mean loss {sum(vals)/len(vals):.4f}   mean cover90 {sum(cov)/len(cov):.3f}   beats persistence in {wins}/{len(rows)}")


def practice_walkforward(kit):
    """Walk-forward over the kit's practice histories: origins at every read from the
    fifth onward, targets at the case's own target."""
    import glob
    import json
    out = []
    for path in sorted(glob.glob(str(kit / "input" / "*.in"))):
        case = json.loads(Path(path).read_text(encoding="utf-8"))
        for k in range(5, len(case["history"])):
            cut = dict(case)
            cut["history"] = case["history"][:k]
            got = make_case_from_cut(cut)
            if got is None:
                continue
            sub, truth = got
            out.append((path.name, k, sub, truth))
    return out


def make_case_from_cut(case):
    """Score a truncated practice case against the case's own target page, which the
    practice cases never change: every truncated forecast is measured against the
    launch-night page it was cut from."""
    hist = case["history"]
    if len(hist) < 8:
        return None
    truth = hist[-1]["rates"]
    return case, truth


if __name__ == "__main__":
    kit = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "workspace/makefun/prep-gta6-trophy-rarity-forecast-01/gta6-trophy-rarity-forecast-01-kit"
    print("practice walk-forward (histories never change: every model must tie with persistence)")
    rows_p = []
    for name, k, sub, truth in practice_walkforward(kit):
        est = mine(sub)
        loss_p = None
        qs_p, qs_m = [], []
        for i, t in enumerate(sub["trophies"]):
            last = sub["history"][-1]["rates"][i]
            qs_p.append(M.persistence_quantiles(last, sub["horizon_minutes"]))
            if est is not None and i < len(est) and est[i] is not None:
                med, sd = est[i]
                qs_m.append([M.expit(med + z * sd) for z in Z])
            else:
                qs_m.append(qs_p[-1])
        loss_p = sum(pinball(q, truth[i]) for i, q in enumerate(qs_p)) / len(qs_p)
        loss_m = sum(pinball(q, truth[i]) for i, q in enumerate(qs_m)) / len(qs_m)
        rows_p.append((name, k, loss_p, loss_m))
    if rows_p:
        n = len(rows_p)
        print(f"  {n} origins; persistence mean {sum(r[2] for r in rows_p)/n:.4f}, "
              f"model mean {sum(r[3] for r in rows_p)/n:.4f}, "
              f"model beats persistence in {sum(1 for r in rows_p if r[3] < r[2]-1e-12)}/{n}")

    print("synthetic week one, walk-forward over 24-60 hour origins")
    rows = run(0, 60)
    summarise(rows)
    print("by horizon (hours to forecast):")
    for dh in (24.0, 48.0, 72.0, 96.0, 108.0):
        sub = [r for r in rows if r["dh"] == dh]
        if sub:
            print(f"  h={dh:5.1f}", end="")
            for name in ("persist", "curve", "model"):
                v = [r[f"loss_{name}"] for r in sub]
                print(f"  {name}={sum(v)/len(v):.3f}", end="")
            print()


def recovery(T_n=45.0):
    """Truth recovery, which does not depend on the synthetic week-one worlds: build a
    case whose truth is a straight line in ln(1 + hours) - the model's own null
    hypothesis, curvature zero - and ask whether the estimator puts the median back
    where it belongs. For each world the tracked count is pinned on 638 -> 3868 at
    5.3 hours and every trophy's own count follows a power law, so the logit rate
    falls by kappa logit units per unit of ln(1 + hours)."""
    rows = []
    for kappa in (0.10, 0.25, 0.50, 0.80, 1.10):
        a_n = _pin_g0(N_FIRST, N_REFRESH, U_REFRESH, T_n, 1.0)
        N = lambda u: N_FIRST * (1.0 + u / T_n) ** a_n
        rates0 = [0.63] + [100.0] * 54
        rates0 = rates0 + [round(0.55 + 0.045 * i, 2) for i in range(53)]
        trophies = []
        for i in range(55):
            if i == 1:
                e = lambda u: N(u)
            else:
                e0 = N_FIRST * rates0[i] / 100.0
                a_e = a_n - kappa
                e = lambda u, e0=e0, a_e=a_e: e0 * (1.0 + u / T_n) ** a_e
            trophies.append({"id": i + 1, "e": e, "e0": rates0[i]})
        w = {"N": N, "trophies": trophies}
        for u_origin in (48.0, 60.0):
            for dh in (48.0, 108.0):
                u_target = u_origin + dh
                got = make_case(w, u_origin, u_target, 0.5)
                if got is None:
                    continue
                case, truth = got
                est = mine(case)
                if est is None:
                    continue
                errs, losses, covers = [], [], []
                for i in range(55):
                    if i == 1:
                        continue
                    if est[i] is None:
                        continue
                    med, sd = est[i]
                    errs.append(abs(med - logit(truth[i])))
                    q = [M.expit(med + z * sd) for z in Z]
                    losses.append(pinball(q, truth[i]))
                    covers.append(logit(q[0]) <= logit(truth[i]) <= logit(q[-1]))
                rows.append((kappa, u_origin, dh,
                            sum(errs) / len(errs), max(errs),
                            sum(losses) / len(losses), sum(covers) / len(covers)))
    return rows


def print_recovery(rows):
    print("truth recovery: straight-line-in-ln(1+hours) truths, curvature zero")
    print("  kappa  origin  h    median err  worst err  loss    cover90")
    for kappa, u_origin, dh, me, we, lo, cov in rows:
        print("  %4.2f  %4.0f   %3.0f   %8.4f   %8.4f  %5.4f   %6.3f"
              % (kappa, u_origin, dh, me, we, lo, cov))
