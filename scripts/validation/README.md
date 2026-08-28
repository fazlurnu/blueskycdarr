# MC-vs-IPS validation campaign at P ~ 1e-4

One script: [`mc_vs_ips_campaign.py`](mc_vs_ips_campaign.py). Every cell of
(pos_ci95 × vel_ci95 × crossing angle) is validated **at the same target
probability** — default 1e-4 — rather than at the same distance: the Monte-Carlo arm
(5,000,000 encounters per cell) keeps the cell's full min-sep distribution, the rare
boundary **d\*** is placed at that cell's own empirical 1e-4 depth quantile (~500
events → a tight 2× anchor), and the IPS ladder descends to d\* on quantile-placed
shells (~0.45 conditional survival each — the recipe validated at the 1e-4 boundary in
`results/ips_mc_comparison/`). The 50 m protected-zone radius is kept as an
intermediate rung, so the classic P(LoS) ratio is reported for free from the same
ladders.

**The flown encounter and its CDR chain**: both drones at 15 m/s, aimed
dead-centre (dcpa 0), spawned *outside* the detection horizon — tlos 150 s against a
120 s lookahead, so the pair approaches ballistically until the predicted CPA enters
the horizon at t ≈ 30 s (the JRESS-style regime; t_max 300 s). Detection is
state-based CPA, resolution is MVP (margin 1.05 → 52.5 m), recovery is
**Probabilistic FTR** (γ = 0.999): a resolving aircraft resumes navigation only when
P(DCPA > rpz) > γ under the worldview uncertainty, for both intruder hypotheses
(ADR 0006). Both arms receive the identical chain.

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

# the server run: 24 cells × 5M encounters, d* at each cell's 1e-4 quantile,
# IPS N=1000 — and raise --reps toward the core count
.venv/bin/python scripts/validation/mc_vs_ips_campaign.py --production --reps 96 --jobs 96
```

Overrides: `--encounters`, `--target-p`, `--particles`, `--reps`, `--jobs`, `--seed`,
`--pos-ci95`, `--vel-ci95`, `--dpsi`.
Deterministic per seed; rerunning reproduces the tables bit for bit.

## Sizing it for the server

- **MC arm** parallelises over episodes and dominates the budget. The long-approach
  regime makes each episode ~4× the short-tlos cost (~180–200 s of simulated flight),
  so expect ~2–2.5k encounters/s on ~100 workers: 120 M encounters (24 × 5 M) ≈
  **13–17 h**. Trim `--encounters` if that must fit a shorter window (1 M/cell ≈ 3 h
  and still ~100 anchor events at 1e-4).
- **IPS arm** parallelises over *replications* (particles within one replication are
  serial), so its ceiling is `min(jobs, reps)`. With `--reps 96` on ~100 cores each
  cell's wall is ≈ one replication → **~8 h at the N=1000 default** over 24 cells
  (~2 h at `--particles 256`, at real risk of ladder degeneracy in this regime —
  the smoke measured mid-ladder conditionals near 0.13, which thin clouds cannot
  carry), with 96 replications per cell of statistical quality.
- Total ≈ **~1 day at the defaults** (MC 13–17 h + IPS ~8 h) on ~100 cores; pad for
  slower cores. Memory: a few MB per worker plus ~40 MB per cell of retained min-sep (~1 GB
  in the parent).

## Outputs (under `results/validation/`)

- `mc_vs_ips.csv` — one row per cell: d\*, both estimates at d\* and at the 50 m rpz,
  both ratios, event counts, collapse count, verdict, wall time.
- `mc_vs_ips.md` — the summary table with the PASS/FAIL/NO_ANCHOR/UNJUDGED tally.
- `mc_vs_ips_detail.jsonl` — per cell: the ladder used and every replication's
  survival fractions (what you read when a cell FAILs or collapses).

## Reading the results

- **PASS / FAIL** — the d\* ratio against ADR-0022-style bands: 2× when the anchor
  has ≥30 events (the design point: ~500 at 1e-4 × 5M), 3× at 10–29, 5× at 1–9.
- **ratio@rpz** — the same comparison at the classic 50 m LoS boundary, read off the
  same ladders as a prefix product; expect it tighter than the d\* ratio (far more
  anchor events, shallower ladder prefix).
- **UNJUDGED** — fewer than 4 IPS replications (the dummy default): ratios shown for
  eyeballing only.
- Collapses (`n_collapsed > 0`) are informative, not errors: at N=256 an occasional
  collapsed replication marks a pinch shell; many mean the cell is drifting toward
  cliff structure — check the JSONL survivals, widen `--particles`, or accept the
  finding.
