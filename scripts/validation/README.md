# MC-vs-IPS validation campaign

One script: [`mc_vs_ips_campaign.py`](mc_vs_ips_campaign.py). Per cell of
(pos_ci95 × vel_ci95 × crossing angle), it estimates the per-encounter P(LoS) twice —
plain Monte Carlo (one declarative sweep, common random numbers across cells) and
fixed-level IPS (ladder placed from that cell's own MC min-sep quantiles, ~0.45
conditional survival per shell) — and judges the ratio in ADR 0022 bands.

## Run

```bash
# smoke the whole pipeline locally (~2 min, 4 cells, tiny budgets — verdicts at these
# budgets exercise the plumbing, not the science; judge agreement at --production)
.venv/bin/python scripts/validation/mc_vs_ips_campaign.py

# the server run (24 cells: pos {10,25,40} × vel {1,3} × dpsi {45,90,135,180};
# 20k MC encounters/cell, IPS N=128 × 8 replications)
.venv/bin/python scripts/validation/mc_vs_ips_campaign.py --production
```

Overrides: `--encounters`, `--particles`, `--reps`, `--jobs`, `--seed`. Everything is
deterministic per seed; rerunning reproduces the tables bit for bit.

## Sizing it for the server

- The **MC arm** parallelises over episodes: `--jobs -1` uses every core, throughput
  ~750 encounters/s on 8 cores (scales with cores).
- The **IPS arm** parallelises over *replications only* — particles within one
  replication are serial through that worker's engine — so its effective parallelism
  is `min(jobs, reps)`. On a 32-core machine, raise `--reps` (e.g. 16–32): more
  replications is also statistically the right dial, since replications, not
  particles, are the unit of spread.
- Rough production wall on 8 cores: MC ≈ 10–12 min (480k encounters), IPS ≈ 10–15 min
  (24 cells × N=128 × 8 reps). Memory is negligible.

## Outputs (under `results/validation/`)

- `mc_vs_ips.csv` — one row per cell: both estimates, event counts, ratio, collapse
  count, verdict, wall time.
- `mc_vs_ips.md` — the summary table with the PASS/FAIL/NO_ANCHOR tally and the
  band definitions.
- `mc_vs_ips_detail.jsonl` — per cell: the ladder used and every replication's
  survival fractions (what you read when a cell FAILs or collapses).

## Reading the results

- **UNJUDGED** — fewer than 4 IPS replications ran (the dummy default): the ratio is
  reported for eyeballing, but with 2 replications the IPS estimate's own spread is
  unquantifiable, so the campaign refuses to call PASS or FAIL on it.
- **PASS** — ratio within the band (2× at ≥30 anchor events, 3× at 10–29, 5× at 1–9).
- **FAIL** — outside the band. Look at the JSONL first: a collapsed replication or a
  pinch shell (one survival ≪ others) usually explains it — widen `--particles` or
  accept that the cell is cliff-structured (see `docs/rare-events.md`).
- **NO_ANCHOR** — MC saw zero events at the rpz boundary within its budget; the cell
  is beyond MC's reach at this budget and the IPS number stands alone. Raising
  `--encounters` may recover an anchor; these are also exactly the cells IPS exists
  for.
- Collapses (`n_collapsed > 0`) are informative, not errors: the ladder told you its
  spacing was too aggressive for that cell's failure structure.

Low-noise cells (pos_ci95 = 10) with good reception sit near the *cliff* regime —
failures there are discrete (never-detected encounters), min-sep is bimodal, and
fixed-level ladders are expected to struggle (collapse or FAIL) at modest N. That is a
finding about the method-cell match, not a bug; `docs/rare-events.md` explains, and
adaptive levels (AMS) are the known remedy if those cells matter.
