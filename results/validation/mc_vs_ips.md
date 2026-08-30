# MC vs IPS validation at target p ~ 0.0001 — 2026-08-28 10:49:45

Seed 42 · MC 1000000/cell · IPS N=1000, 100 reps · comm rx=0.8 lat=0.0s · dcpa 0, tlos 150s, lookahead 120s, rpz 50m · state-based CD, MVP (margin 1.05), ProbFTR (gamma 0.999) · d* at each cell's 0.0001 depth quantile

**9 PASS / 0 FAIL / 0 NO_ANCHOR / 0 UNJUDGED** of 9 cells · MC arm 5321s · total 13373s

| dpsi | pos | vel | d* [m] | P_MC(d*) | events | P_IPS(d*) | ratio | ratio@rpz | collapsed | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 45 | 3 | 1 | 50 | 9.7e-05 | 97 | 0.000109 | 1.12 | 1.12 | 3/100 | PASS |
| 45 | 10 | 1 | 51.3 | 9e-05 | 90 | 8.26e-05 | 0.92 | nan | 10/100 | PASS |
| 45 | 30 | 1 | 55.2 | 9.8e-05 | 98 | 0.000118 | 1.20 | nan | 2/100 | PASS |
| 90 | 3 | 1 | 50.1 | 9.7e-05 | 97 | 8.17e-05 | 0.84 | nan | 77/100 | PASS |
| 90 | 10 | 1 | 51.9 | 9.7e-05 | 97 | 9.54e-05 | 0.98 | nan | 75/100 | PASS |
| 90 | 30 | 1 | 56.9 | 9.6e-05 | 96 | 0.000133 | 1.39 | nan | 76/100 | PASS |
| 180 | 3 | 1 | 49.7 | 9.9e-05 | 99 | 0.000151 | 1.52 | nan | 88/100 | PASS |
| 180 | 10 | 1 | 50.9 | 9.4e-05 | 94 | 0.000174 | 1.85 | nan | 89/100 | PASS |
| 180 | 30 | 1 | 56.8 | 9.9e-05 | 99 | 7.47e-05 | 0.75 | nan | 90/100 | PASS |

Primary verdicts at d* (the target-probability boundary); the rpz ratio is the classic P(LoS) read off the same ladders as a prefix product. Bands (ADR 0022 style): 2x at >=30 anchor events, 3x for 10-29, 5x for 1-9; NO_ANCHOR when MC saw none; UNJUDGED below 4 replications (smoke budgets).
