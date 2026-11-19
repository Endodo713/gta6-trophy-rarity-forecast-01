# gta6-trophy-rarity-forecast-01

Forecaster for the make.fun competition `gta6-trophy-rarity-forecast-01`:
forecast every Grand Theft Auto VI trophy's Exophase earn rate at
2026-11-26 00:00 UTC as five quantiles (5/25/50/75/95%), scored by
pinball loss on the logit scale.

## Model

Per trophy, a weighted least-squares fit of the logit earn rate against
`x = ln(1 + hours since first read)`, evaluated at the case target:

- **Quadratic per trophy** (`POOL_CURVATURE = False`): each trophy fits its
  own `[1, x, x^2]` curve — pooled curvature from the players/earned totals
  was tested and lost at every curvature scale.
- **Recency weighting** (`TAU_AGE = 6.0` h): reads decay with a 6-hour
  half-life-ish scale. Swept over {inf, 72, 48, 24, 12, 6, 2}; shorter
  windows win monotonically because the trend is curved in `ln(1+hours)`
  and the log-spaced read grid is dense near the end.
- **No thinning** (`X_KNOT = 0.0`): evenly-spaced x-knot subsampling hurt at
  every curvature scale, so all points are kept.
- **Spread**: fit covariance at the target plus a structural floor
  (`STRUCTURAL = 0.10` logit, the loss-optimum in the recovery test;
  pinball penalises an over-wide band more than a narrow one).
- **Fallbacks, layered**: per-trophy fit failure → pooled log-ratio trend →
  persistence baseline (hold the last rate, random-walk spread). The program
  always writes a valid output.

Standard library only, offline, deterministic, no network, no clock, no
randomness.

## How to run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

Reads every `*.in` in the input directory, writes one `<id>.out` per case:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`,
55 rows, each quantile finite, 0–100, non-decreasing.

## Backtest results

Rolling-origin / walk-forward on synthetic week-one worlds (60 seeds ×
2 refresh regimes × 3 origins × 5 horizons = 1800 cases), pinball loss
on the logit scale, full run in `scripts/backtest.py`
(`python3 scripts/backtest.py`):

| method | mean loss | mean cover90 | beats persistence |
|---|---|---|---|
| persistence (house starter) | 0.6568 | 0.837 | 0/1800 |
| curve (house trend baseline) | 0.4701 | 0.901 | 1787/1800 |
| **this model** | **0.3314** | 0.844 | **1800/1800** |

By horizon (hours ahead):

| h | persistence | curve | model |
|---|---|---|---|
| 24 | 0.413 | 0.311 | 0.181 |
| 48 | 0.582 | 0.412 | 0.275 |
| 72 | 0.692 | 0.488 | 0.347 |
| 96 | 0.780 | 0.555 | 0.413 |
| 108 | 0.817 | 0.584 | 0.441 |

Recovery test (truth exactly the model's own null hypothesis): median
logit error 0.079–0.135 across curvature scales, vs 0.64–1.03 for the
unweighted pooled version.

## Practice verification

`p00.in` and `p01.in` from the kit:

- output valid: 55 rows, ids 1–55, 5 finite quantiles 0–100,
  non-decreasing, 2,628 bytes per file (limit 65,536);
- deterministic: two runs byte-identical;
- runtime ~0.09 s per case (limit 8 s).

## Honest notes

- The practice histories are totally flat (every read identical), so on
  practice the model ties persistence — medians identical, quantiles
  differing only in the 4th decimal from rounding. Practice cannot
  validate the model; the design targets the final case (horizon ~4.54
  days, ~2.5 days of history including post-refresh reads).
- Practice scores cannot be measured locally: the practice answers are
  held by the house and are not in the kit. No practice score is claimed.
- The synthetic backtest worlds are anchored on the observed
  launch-night → refresh drop (players tracked 638 → 3,868, rates
  falling), with an acceptance filter on plausible week-one magnitudes.
  They are simulations, not the real final data; the final outcome
  depends on the real Exophase reads.
