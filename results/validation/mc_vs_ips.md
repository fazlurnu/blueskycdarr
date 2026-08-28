# MC vs IPS validation at target p ~ 0.002 — 2026-08-28 09:04:30

Seed 42 · MC 4000/cell · IPS N=24, 2 reps · comm rx=0.8 lat=0.3s · dcpa 0, tlos 150s, lookahead 120s, rpz 50m · state-based CD, MVP (margin 1.05), ProbFTR (gamma 0.999) · d* at each cell's 0.002 depth quantile

**0 PASS / 0 FAIL / 0 NO_ANCHOR / 1 UNJUDGED** of 1 cells · MC arm 26s · total 34s

| dpsi | pos | vel | d* [m] | P_MC(d*) | events | P_IPS(d*) | ratio | ratio@rpz | collapsed | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 90 | 40 | 3 | 14.5 | 0.00175 | 7 | 0 | 0.00 | 0.04 | 2/2 | UNJUDGED |

Primary verdicts at d* (the target-probability boundary); the rpz ratio is the classic P(LoS) read off the same ladders as a prefix product. Bands (ADR 0022 style): 2x at >=30 anchor events, 3x for 10-29, 5x for 1-9; NO_ANCHOR when MC saw none; UNJUDGED below 4 replications (smoke budgets).
