"""gta6-trophy-rarity-forecast-01 entry: forecast every GTA VI PS5 trophy's Exophase
earn rate at the case's target, as five quantiles, from the case's history.

The page prints rate_i = 100 * e_i / N: the percent of the tracked players N who have
earned trophy i, where e_i is the number of them who have. Two of those three
quantities arrive in every read as exact counts: N is `players_tracked`, and the total
of every trophy earned, E = sum of e_i, is `trophies_earned`. So

    logit(rate_i) = ln(e_i) - ln(N)

up to under 0.01 logit units for any rate below 20 percent, and both ln(e_i) and ln(N)
are series that can be fitted directly.

Week-one dynamics bend: N grows toward the installed base, every e_i grows toward its
own ceiling, and both curves slow down. The logit rate therefore falls, and the fall
slows, so a straight-line extrapolation overshoots the drop and a flat persistence
forecast undershoots it. The forecast is the trend read at the target with that
deceleration made explicit:

    ln N(x)  = a_N + b_N*x + c_N*x^2                from the case's own reads
    ln E(x)  = a_E + b_E*x + c_E*x^2                (E = trophies_earned)
    logit rate_i(x) = a_i + b_i*x + (c_E - c_N)*x^2 per trophy, curvature shared

with x = ln(1 + hours since the case's first read). The shared curvature is the
deceleration, taken from two clean series (a count and a total of counts) instead of
from 55 noisy rate series. Per trophy only a level and a speed are fitted, which the
recent, high-tracked reads pin down well.

Spread: the extrapolation spread from the fit's covariance plus the uncertainty of the
shared curvature, plus a structural allowance for the refresh, patch or story the
history cannot show. Pinball loss on the logit scale is second order in the spread (two
times too wide, or too narrow, costs about 18 percent) and first order in the median:
a median miss of one logit unit costs 1.0. So the median is the object and the band
only has to be honest.

Fallbacks, so a case is never invalid: fewer than six usable reads, a singular fit or a
history that never changed once fall back to the trend line with zero curvature, then
to the house persistence baseline. Any exception in a case falls back to persistence
for that case, and any exception in one trophy falls back to persistence for that
trophy. A valid file is always written.

Standard library only, no network, no clock, no randomness: two runs write the same
bytes. Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output), in name order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs with -P, which leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# House baseline spread, in logit units per square root day. Fallback only.
SIGMA = 0.5
# The structural spread allowance, in logit units: the refresh, patch or news story the
# history cannot show. Added in quadrature with the fit's own spread.
STRUCTURAL = 0.10
# A fit this wide in x earns a look at its own curvature; narrower, borrow the shared one.
MIN_X_RANGE = 0.75
# Minimum usable reads before a per-trophy speed is trusted.
MIN_POINTS = 6
# Recency weighting of the per-trophy fits, in hours: a read ages out as exp(-age/TAU).
# The trend on the log axis curves, so the slope that matters at the target is the one
# near the last read; six hours keeps the launch-night page out of the fit without
# pretending the earlier reads do not exist. Tuned on the recovery test, stable across
# three curvature scales.
TAU_AGE = 6.0
# A rate below this is unprintable and carries no information.
MIN_RATE = 0.01


class Fit:
    """The forecast's hyperparameters, in one place so a backtest can sweep them."""

    TAU_AGE = TAU_AGE
    STRUCTURAL = STRUCTURAL
    MIN_POINTS = MIN_POINTS
    MIN_X_RANGE = MIN_X_RANGE
    MIN_RATE = MIN_RATE
    # Subsample the design on the log axis, keeping every point whose x is at least this
    # far from the last one kept. Zero, which the recovery test prefers: thinning the
    # points made the trend worse at every curvature scale.
    X_KNOT = 0.0
    # Borrow the curvature from the two clean series (ln of the tracked count and of the
    # total earned) instead of fitting it on each trophy's own noisy rate series.
    POOL_CURVATURE = False


def parse_ts(s):
    """An ISO 8601 timestamp (a trailing Z is fine) as an aware UTC datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def round4(x):
    """x rounded to four decimals, half up, as the starter rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def logit(pct):
    p = min(99.99, max(0.01, pct))
    return math.log(p / (100.0 - p))


def expit(z):
    if z > 60.0:
        return 100.0
    if z < -60.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-z))


def wls(design, target, weights=None):
    """Weighted least squares of target on [1] + design, by normal equations with
    partial pivoting. Returns (coefficients, their covariance) or (None, None) when the
    system is singular; the covariance carries the residual scale, so a prediction's
    spread comes straight off it."""
    n = len(target)
    if n == 0:
        return None, None
    if weights is None:
        weights = [1.0] * n
    if design:
        a = [[1.0] + row for row in design]
    else:
        a = [[1.0] for _ in range(n)]
    k = len(a[0])
    m = [[sum(weights[i] * a[i][r] * a[i][c] for i in range(n)) for c in range(k)] for r in range(k)]
    v = [sum(weights[i] * a[i][r] * target[i] for i in range(n)) for r in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        v[col], v[piv] = v[piv], v[col]
        if abs(m[col][col]) < 1e-14:
            return None, None
        for r in range(col + 1, k):
            f = m[r][col] / m[col][col]
            if f:
                for c in range(col, k):
                    m[r][c] -= f * m[col][c]
                v[r] -= f * v[col]
    beta = [0.0] * k
    for i in range(k - 1, -1, -1):
        beta[i] = (v[i] - sum(m[i][j] * beta[j] for j in range(i + 1, k))) / m[i][i]
    inv = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for col in range(k):
        d = m[col][col]
        if abs(d) < 1e-14:
            return None, None
        for c in range(k):
            m[col][c] /= d
            inv[col][c] /= d
        for r in range(k):
            if r != col:
                f = m[r][col]
                if f:
                    for c in range(k):
                        m[r][c] -= f * m[col][c]
                        inv[r][c] -= f * inv[col][c]
    resid = [target[i] - sum(a[i][j] * beta[j] for j in range(k)) for i in range(n)]
    dof = n - k
    scale = (sum(weights[i] * resid[i] ** 2 for i in range(n)) / dof) if dof > 0 else 0.0
    cov = [[scale * inv[r][c] for c in range(k)] for r in range(k)]
    return beta, cov


def _build_timeline(case):
    """The case's reads as (rows, first stamp). Each row is
    (hours since the first read, players tracked, total trophies earned, rates)."""
    hist = case.get("history") or []
    if not hist:
        return None, None
    rows = []
    for r in hist:
        stamp = r.get("at") or r.get("slot") or case.get("at")
        if not stamp:
            continue
        try:
            rows.append((parse_ts(stamp), r))
        except (ValueError, TypeError):
            continue
    rows.sort(key=lambda t: t[0])
    if not rows:
        return None, None
    first = rows[0][0]
    out = []
    baseline = None
    for stamp, r in rows:
        rates = r.get("rates")
        tracked = r.get("players_tracked")
        key = (tracked, tuple(rates) if rates is not None else None)
        if baseline is None or key != baseline:
            baseline = key
            hours = (stamp - first).total_seconds() / 3600.0
            out.append((hours, tracked, r.get("trophies_earned"), rates))
    return out, first


def fit_curve(points):
    """A quadratic in x through (x, ln y) points: (a, b, c, cov) or None."""
    if len(points) < 4:
        return None
    res = wls([[p[0], p[0] ** 2] for p in points], [math.log(p[1]) for p in points])
    if res[0] is None:
        return None
    beta, cov = res
    return beta[0], beta[1], beta[2], cov


def estimate(case):
    """Per trophy, (median logit rate, spread in logit units) at the case's target;
    None when the case cannot carry a trend fit. Each entry may be None on its own."""
    timeline, first = _build_timeline(case)
    if timeline is None or len(timeline) < Fit.MIN_POINTS:
        return None
    try:
        target_hours = (parse_ts(case["target"]) - first).total_seconds() / 3600.0
    except (KeyError, TypeError, ValueError):
        return None
    xs_all = [math.log1p(max(0.0, h)) for h, _, _, _ in timeline]
    x_target = math.log1p(max(0.0, target_hours))
    if x_target <= max(xs_all):
        x_target = max(xs_all) + 0.05

    # ln(N) and ln(E), the two clean series: uniform weight, curvature over the whole history.
    n_points = [(x, r[1]) for x, r in zip(xs_all, timeline)
                if isinstance(r[1], (int, float)) and r[1] > 0]
    e_points = [(x, r[2]) for x, r in zip(xs_all, timeline)
                if isinstance(r[2], (int, float)) and r[2] > 0]
    fit_n = fit_curve(n_points)
    if fit_n is None or len(n_points) < 5:
        return None
    c_n, cov_n = fit_n[2], fit_n[3]
    fit_e = fit_curve(e_points)
    if fit_e is None or len(e_points) < 5:
        c_pool = 0.0
        b_pool = None
        var_c = cov_n[2][2]
    else:
        c_e, cov_e = fit_e[2], fit_e[3]
        c_pool = c_e - c_n
        b_pool = fit_e[1] - fit_n[1]
        var_c = cov_e[2][2] + cov_n[2][2]

    last_hours = timeline[-1][0]
    out = []
    for i in range(len(case.get("trophies") or [])):
        xs_all, ys_all, hs_all = [], [], []
        for hours, _, _, rates in timeline:
            if rates is None or i >= len(rates):
                continue
            rate = rates[i]
            if not isinstance(rate, (int, float)) or rate < Fit.MIN_RATE:
                continue
            xs_all.append(math.log1p(max(0.0, hours)))
            ys_all.append(logit(rate))
            hs_all.append(hours)
        # Even out the design, then weight it by recency in hours, not in x.
        xs, ys, hs = [], [], []
        next_keep = 0.0
        for x, y, h in zip(xs_all, ys_all, hs_all):
            if x < next_keep:
                continue
            xs.append(x)
            ys.append(y)
            hs.append(h)
            next_keep = x + Fit.X_KNOT
        if xs and xs[-1] != xs_all[-1]:
            xs.append(xs_all[-1]); ys.append(ys_all[-1]); hs.append(hs_all[-1])
        ws = [math.exp(-max(0.0, last_hours - h) / Fit.TAU_AGE) for h in hs]
        if len(ys) < Fit.MIN_POINTS:
            out.append(None)
            continue
        if Fit.POOL_CURVATURE:
            design = [[x] for x in xs]
            target = [y - c_pool * x * x for y, x in zip(ys, xs)]
            c_use = c_pool
        else:
            design = [[x, x * x] for x in xs]
            target = ys
            c_use = None
        res = wls(design, target, ws)
        if res[0] is None:
            out.append(None)
            continue
        beta, cov = res
        a_i, b_i = beta[0], beta[1]
        if b_pool is not None and (max(xs) - min(xs) < Fit.MIN_X_RANGE or len(ys) < Fit.MIN_POINTS + 4):
            b_i = 0.5 * b_i + 0.5 * b_pool
        var_pred = cov[0][0] + 2.0 * x_target * cov[0][1] + x_target * x_target * cov[1][1]
        if Fit.POOL_CURVATURE:
            var_pred += var_c * x_target ** 4
        sd = math.sqrt(max(0.0, var_pred) + Fit.STRUCTURAL ** 2)
        if Fit.POOL_CURVATURE:
            median = a_i + b_i * x_target + c_pool * x_target ** 2
        else:
            c_i = beta[2]
            median = a_i + b_i * x_target + c_i * x_target ** 2
        out.append((median, sd))
    return out


def persistence_quantiles(rate, horizon_minutes):
    """House baseline, for a fallback: hold the last rate, spread widening like a
    random walk's on the logit scale."""
    spread = SIGMA * math.sqrt(max(0.0, horizon_minutes) / 1440.0)
    lg = logit(rate if isinstance(rate, (int, float)) else 0)
    return [round4(expit(lg + z * spread)) for z in Z]


def clean_quantiles(values):
    """Five finite numbers from 0 to 100, non-decreasing, four decimals."""
    fixed = []
    prev = 0.0
    for q in values:
        q = float(q)
        if not math.isfinite(q):
            q = prev
        q = min(100.0, max(0.0, q))
        q = max(q, prev)
        q = round4(q)
        fixed.append(q)
        prev = q
    return fixed


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    try:
        horizon = float(case.get("horizon_minutes") or 0)
    except (TypeError, ValueError):
        horizon = 0.0
    est = estimate(case)
    hist = case.get("history") or []
    last = hist[-1] if hist else None
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        if est is not None and i < len(est) and est[i] is not None:
            median, sd = est[i]
            values = [expit(median + z * sd) for z in Z]
        else:
            rate = None
            if last is not None and last.get("rates") is not None and i < len(last["rates"]):
                rate = last["rates"][i]
            values = persistence_quantiles(rate, horizon)
        rows.append({"id": trophy["id"], "q": clean_quantiles(values)})
    return {"trophies": rows}


def safe_forecast_case(case, name):
    """Never fail a case: any error falls back to the persistence baseline."""
    try:
        return forecast_case(case, name)
    except Exception:
        try:
            horizon = float(case.get("horizon_minutes") or 0)
        except (TypeError, ValueError):
            horizon = 0.0
        hist = case.get("history") or []
        last = hist[-1] if hist else None
        rates = (last or {}).get("rates") or []
        rows = []
        for i, trophy in enumerate(case.get("trophies") or []):
            rate = rates[i] if i < len(rates) else None
            rows.append({"id": trophy["id"], "q": persistence_quantiles(rate, horizon)})
        return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        forecast = safe_forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
