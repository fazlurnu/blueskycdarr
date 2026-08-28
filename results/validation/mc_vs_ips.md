# MC vs IPS validation at target p ~ 0.002 — 2026-08-28 06:35:17

Seed 42 · MC 20000/cell · IPS N=48, 2 reps · comm rx=0.8 lat=0.3s · dcpa 0, tlos 20s, rpz 50m · d* at each cell's 0.002 depth quantile

**0 PASS / 0 FAIL / 0 NO_ANCHOR / 4 UNJUDGED** of 4 cells · MC arm 82s · total 104s

| dpsi | pos | vel | d* [m] | P_MC(d*) | events | P_IPS(d*) | ratio | ratio@rpz | collapsed | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 90 | 25 | 3 | 10.8 | 0.00195 | 39 | 0.00489 | 2.51 | 2.20 | 1/2 | UNJUDGED |
| 90 | 40 | 3 | 5.5 | 0.0019 | 38 | 0.000452 | 0.24 | 0.95 | 1/2 | UNJUDGED |
| 135 | 25 | 3 | 8 | 0.00195 | 39 | 0.014 | 7.17 | 1.52 | 1/2 | UNJUDGED |
| 135 | 40 | 3 | 6.1 | 0.0019 | 38 | 0.00262 | 1.38 | 0.63 | 0/2 | UNJUDGED |

Primary verdicts at d* (the target-probability boundary); the rpz ratio is the classic P(LoS) read off the same ladders as a prefix product. Bands (ADR 0022 style): 2x at >=30 anchor events, 3x for 10-29, 5x for 1-9; NO_ANCHOR when MC saw none; UNJUDGED below 4 replications (smoke budgets).
