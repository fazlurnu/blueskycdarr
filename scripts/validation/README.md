# MC-vs-IPS validation campaign at P ~ 1e-4

One script: [`mc_vs_ips_campaign.py`](mc_vs_ips_campaign.py). Every cell of
(pos_ci95 × vel_ci95 × crossing angle) is validated **at the same target
probability** — default 1e-4 — rather than at the same distance: the Monte-Carlo arm
(1,000,000 encounters per cell) keeps the cell's full min-sep distribution, the rare
boundary **d\*** is placed at that cell's own empirical 1e-4 depth quantile (~100
events → a tight 2× anchor), and the IPS ladder descends to d\* on quantile-placed
shells (~0.45 conditional survival each — the recipe validated at the 1e-4 boundary in
`results/ips_mc_comparison/`). The 50 m protected-zone radius is kept as an
intermediate rung, so the classic P(LoS) ratio is reported for free from the same
ladders.

**Why the boundary moves, not the physics**: in this CDR stack the graded failure
family bottoms out near P(LoS) ~ 1e-3; dialling the cell physics rarer flips the
failure mechanism to a discrete, bimodal *cliff* where fixed-level ladders collapse
(see `docs/rare-events.md`). Depth of breach stays graded in every cell, so the 1e-4
event that is actually ladderable is a deep breach — and that is what d\* measures.
For the same reason the default grid keeps the graded corner (pos_ci95 ∈ {25, 40} m,
vel_ci95 ∈ {1, 3} m/s, dpsi ∈ {45, 90, 135, 180}°, 16 cells); strong-CDR cells
(pos_ci95 ≈ 10 m) are cliff-structured and would need AMS, not a finer fixed ladder.

## Run

```bash
# smoke the whole pipeline locally (~3 min, 4 cells, 20k encounters, target 2e-3 —
# verdicts at these budgets are UNJUDGED by design; judge at --production)
.venv/bin/python scripts/validation/mc_vs_ips_campaign.py

# the server run: 16 cells × 1M encounters, d* at each cell's 1e-4 quantile,
# IPS N=256 — and raise --reps toward the core count
.venv/bin/python scripts/validation/mc_vs_ips_campaign.py --production --reps 96 --jobs 96
```

Overrides: `--encounters`, `--target-p`, `--particles`, `--reps`, `--jobs`, `--seed`.
Deterministic per seed; rerunning reproduces the tables bit for bit.

## Sizing it for the server

- **MC arm** parallelises over episodes and dominates the budget: 16 M encounters at
  ~8–10k encounters/s on ~100 workers ≈ **25–35 min**.
- **IPS arm** parallelises over *replications* (particles within one replication are
  serial), so its ceiling is `min(jobs, reps)`. With `--reps 96` on ~100 cores each
  cell's wall is ≈ one replication (~60–90 s for a ~12-shell ladder at N=256) →
  **~20–30 min**, with 96 replications per cell of statistical quality.
- Total ≈ **1 hour** on ~100 cores; pad ~50% for slower server cores. Memory is a few
  MB per worker plus ~8 MB per cell of retained min-sep data.

## Outputs (under `results/validation/`)

- `mc_vs_ips.csv` — one row per cell: d\*, both estimates at d\* and at the 50 m rpz,
  both ratios, event counts, collapse count, verdict, wall time.
- `mc_vs_ips.md` — the summary table with the PASS/FAIL/NO_ANCHOR/UNJUDGED tally.
- `mc_vs_ips_detail.jsonl` — per cell: the ladder used and every replication's
  survival fractions (what you read when a cell FAILs or collapses).

## Reading the results

- **PASS / FAIL** — the d\* ratio against ADR-0022-style bands: 2× when the anchor
  has ≥30 events (the design point: ~100 at 1e-4 × 1M), 3× at 10–29, 5× at 1–9.
- **ratio@rpz** — the same comparison at the classic 50 m LoS boundary, read off the
  same ladders as a prefix product; expect it tighter than the d\* ratio (far more
  anchor events, shallower ladder prefix).
- **UNJUDGED** — fewer than 4 IPS replications (the dummy default): ratios shown for
  eyeballing only.
- Collapses (`n_collapsed > 0`) are informative, not errors: at N=256 an occasional
  collapsed replication marks a pinch shell; many mean the cell is drifting toward
  cliff structure — check the JSONL survivals, widen `--particles`, or accept the
  finding.
