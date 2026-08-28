# MC vs IPS validation — 2026-08-28 05:58:37

Seed 42 · MC 3000/cell · IPS N=48, 2 reps · comm rx=0.8 lat=0.3s · dcpa 0, tlos 20s, rpz 50m

**0 PASS / 0 FAIL / 0 NO_ANCHOR / 4 UNJUDGED** of 4 cells · MC arm 17s · total 35s

| dpsi | pos_ci95 | vel_ci95 | P(MC) | events | P(IPS) | ratio | collapsed | verdict |
|---|---|---|---|---|---|---|---|---|
| 90 | 25 | 3 | 0.0103 | 31 | 0.0211 | 2.04 | 1/2 | UNJUDGED |
| 90 | 40 | 3 | 0.021 | 63 | 0.0135 | 0.64 | 0/2 | UNJUDGED |
| 135 | 25 | 3 | 0.0137 | 41 | 0.0296 | 2.17 | 0/2 | UNJUDGED |
| 135 | 40 | 3 | 0.02 | 60 | 0.0141 | 0.70 | 0/2 | UNJUDGED |

Verdict bands (ADR 0022 style): ratio within 2x when the anchor has >=30 events, 3x for 10-29, 5x for 1-9; NO_ANCHOR when MC saw none; UNJUDGED when IPS ran fewer than 4 replications (smoke budgets).
